//! Dynamic Nested Sampling engine.
//!
//! The implementation here provides a modular Dynamic Nested Sampling (DNS)
//! engine that cooperates with Chronopt's [`Sampler`] trait. Behaviour is split
//! across dedicated submodules (state, proposals, scheduler, results) so each
//! concern can evolve independently while preserving a consistent public API.

use super::{evaluate, Sampler, Samples};
use crate::problem::Problem;
use rand::prelude::*;
use rand::rngs::StdRng;

mod proposals;
mod results;
mod scheduler;
mod state;

pub use results::{NestedSample, NestedSamples};

const DEFAULT_LIVE_POINTS: usize = 64;
pub(super) const MIN_LIVE_POINTS: usize = 8;
const DEFAULT_EXPANSION_FACTOR: f64 = 0.5;
const DEFAULT_TERMINATION_TOL: f64 = 1e-3;
const MAX_ITERATION_MULTIPLIER: usize = 1024;

/// Configurable Dynamic Nested Sampling engine that integrates with [`Sampler`].
#[derive(Clone, Debug)]
pub struct DynamicNestedSampler {
    live_points: usize,
    expansion_factor: f64,
    termination_tol: f64,
    parallel: bool,
    seed: Option<u64>,
}

/// Builder-style configuration and execution entry points for [`DynamicNestedSampler`].
impl DynamicNestedSampler {
    /// Create a sampler with default live-point budget and tolerances.
    pub fn new() -> Self {
        Self {
            live_points: DEFAULT_LIVE_POINTS,
            expansion_factor: DEFAULT_EXPANSION_FACTOR,
            termination_tol: DEFAULT_TERMINATION_TOL,
            parallel: false,
            seed: None,
        }
    }

    /// Override the number of live points, clamping to the algorithm minimum.
    pub fn with_live_points(mut self, live_points: usize) -> Self {
        self.live_points = live_points.max(MIN_LIVE_POINTS);
        self
    }

    /// Adjust how aggressively the live set expands when the posterior is broad.
    pub fn with_expansion_factor(mut self, expansion_factor: f64) -> Self {
        self.expansion_factor = expansion_factor.max(0.0);
        self
    }

    /// Set the threshold for evidence convergence that drives termination.
    pub fn with_termination_tolerance(mut self, tolerance: f64) -> Self {
        self.termination_tol = tolerance.abs().max(1e-8);
        self
    }

    /// Opt into parallel proposal generation when supported by the backend.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Fix the RNG seed for reproducible sampling runs.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run the Dynamic Nested Sampling loop starting from the supplied position.
    pub fn run_nested(&self, problem: &Problem, mut initial: Vec<f64>) -> NestedSamples {
        let dimension = match (problem.dimension(), initial.len()) {
            (d, _) if d > 0 => d,
            (0, len) if len > 0 => len,
            _ => 1,
        };

        if initial.len() < dimension {
            initial.resize(dimension, 0.0);
        } else if initial.len() > dimension {
            initial.truncate(dimension);
        }

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                StdRng::from_rng(rand::thread_rng()).unwrap_or_else(|_| StdRng::seed_from_u64(0))
            }
        };

        let bounds = state::Bounds::from_problem(problem, &initial, dimension);
        let live_points = state::initial_live_points(
            problem,
            &bounds,
            &mut rng,
            self.live_points,
            self.expansion_factor,
        );

        if live_points.len() < MIN_LIVE_POINTS {
            return NestedSamples::degenerate(initial);
        }

        let mut sampler_state = state::SamplerState::new(live_points);
        let proposal_engine =
            proposals::ProposalEngine::new(sampler_state.dimension(), self.expansion_factor);
        let scheduler = scheduler::Scheduler::new(
            self.live_points,
            self.expansion_factor,
            self.termination_tol,
        );

        let max_iterations = MAX_ITERATION_MULTIPLIER
            .saturating_mul(self.live_points)
            .saturating_mul(sampler_state.dimension().max(1));

        let config = RunLoopConfig {
            parallel: self.parallel,
            max_iterations,
        };

        run_loop(
            problem,
            &bounds,
            &mut sampler_state,
            proposal_engine,
            scheduler,
            config,
            &mut rng,
        )
        .unwrap_or_else(|state| NestedSamples::degenerate_with_state(initial, state))
    }
}

impl Default for DynamicNestedSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for DynamicNestedSampler {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> Samples {
        self.run_nested(problem, initial).to_samples()
    }
}

#[derive(Clone, Copy)]
struct RunLoopConfig {
    parallel: bool,
    max_iterations: usize,
}

/// Execute the adaptive live-set loop, returning posterior samples on success.
fn run_loop(
    problem: &Problem,
    bounds: &state::Bounds,
    state: &mut state::SamplerState,
    mut proposals: proposals::ProposalEngine,
    mut scheduler: scheduler::Scheduler,
    config: RunLoopConfig,
    rng: &mut StdRng,
) -> Result<NestedSamples, state::SamplerState> {
    let mut iterations = 0usize;

    while iterations < config.max_iterations {
        iterations += 1;

        let info_estimate = results::information_estimate(state.posterior());
        let target_live = scheduler.target(info_estimate, state.live_point_count());
        state.adjust_live_set(target_live);

        if state.live_point_count() < target_live {
            let mut attempts = 0usize;
            let max_attempts = target_live.saturating_mul(16).max(128);

            while state.live_point_count() < target_live && attempts < max_attempts {
                attempts = attempts.saturating_add(1);

                let threshold = state.min_log_likelihood();
                if let Some(new_point) = proposals.draw(
                    rng,
                    problem,
                    state.live_points(),
                    bounds,
                    threshold,
                    config.parallel,
                ) {
                    state.insert_live_point(new_point);
                } else {
                    break;
                }
            }
        }

        if state.live_points().is_empty() {
            break;
        }

        let worst_index = match state.worst_index() {
            Some(idx) => idx,
            None => break,
        };

        let removed = match state.remove_at(worst_index) {
            Some(value) => value,
            None => break,
        };
        let threshold = removed.log_likelihood();

        if let Some(new_point) = proposals.draw(
            rng,
            problem,
            state.live_points(),
            bounds,
            threshold,
            config.parallel,
        ) {
            state.accept_removed(removed);
            state.insert_live_point(new_point);
        } else {
            state.restore_removed(removed);
        }

        if scheduler.should_terminate(state, info_estimate) {
            break;
        }
    }

    if iterations >= config.max_iterations {
        return Err(state.clone());
    }

    state.finalize();

    Ok(results::NestedSamples::build(
        state.posterior(),
        state.dimension(),
    ))
}

/// Compute `log(exp(a) - exp(b))` while guarding against catastrophic cancellation.
pub(super) fn logspace_sub(a: f64, b: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }

    if b > a {
        return None;
    }

    if (a - b).abs() < f64::EPSILON {
        return Some(f64::NEG_INFINITY);
    }

    // Use expm1 for numerical stability: log(exp(a) - exp(b)) = a + log(1 - exp(b-a))
    let diff = -(b - a).exp_m1();
    if diff <= 0.0 {
        return Some(f64::NEG_INFINITY);
    }

    Some(a + diff.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::builders::BuilderParameterExt;
    use crate::problem::{ParameterSpec, ScalarProblemBuilder};
    fn gaussian_problem(mean: f64, sigma: f64) -> crate::problem::Problem {
        let log_norm = sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
        ScalarProblemBuilder::new()
            .with_objective(move |x: &[f64]| {
                let diff = x[0] - mean;
                0.5 * (diff * diff) / (sigma * sigma) + log_norm
            })
            .with_parameter(ParameterSpec::new(
                "x",
                mean,
                Some((mean - 10.0, mean + 10.0)),
            ))
            .build()
            .expect("failed to build gaussian problem")
    }

    #[test]
    fn dynamic_nested_gaussian_behaves_reasonably() {
        let problem = gaussian_problem(1.5, 0.4);
        let sampler = DynamicNestedSampler::new()
            .with_live_points(128)
            .with_expansion_factor(0.2)
            .with_termination_tolerance(2e-4)
            .with_seed(7);

        let nested = sampler.run_nested(&problem, vec![1.5]);

        assert!(nested.draws() > 0, "expected posterior samples");
        assert!(nested.log_evidence().is_finite());
        assert!(nested.information().is_finite());
        let mean = nested.mean()[0];
        assert!(
            mean.is_finite(),
            "posterior mean must be finite, got {:.4}",
            mean
        );

        // The posterior should be concentrated within the prior bounds supplied in the builder.
        assert!(((1.5 - 10.0)..=(1.5 + 10.0)).contains(&mean));

        assert_eq!(nested.posterior().len(), nested.draws());
        assert!(nested
            .posterior()
            .iter()
            .all(|sample| sample.log_likelihood.is_finite() && sample.log_weight.is_finite()));

        let evidence_sum: f64 = nested
            .posterior()
            .iter()
            .map(|sample| sample.evidence_weight())
            .sum();
        assert!(evidence_sum.is_finite() && evidence_sum > 0.0);
    }

    #[test]
    fn logspace_sub_basic() {
        // Test basic functionality: log(exp(5) - exp(3)) = log(exp(5) * (1 - exp(-2)))
        let result = logspace_sub(5.0, 3.0).unwrap();
        let expected = 5.0 + (1.0 - (-2.0_f64).exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn logspace_sub_near_equal() {
        // Test numerical stability when a ≈ b
        let a = 10.0;
        let b = 10.0 - 1e-8;
        let result = logspace_sub(a, b).unwrap();

        // Should be finite and less than a (since we're subtracting)
        assert!(result.is_finite());
        assert!(result < a);

        // For small differences, log(exp(a) - exp(b)) ≈ log(exp(a) * (a-b)) = a + log(a-b)
        // But we need to verify the actual computation is stable
        // The key is that it doesn't produce NaN or infinity
        let diff = a - b;
        assert!(diff > 0.0);
    }

    #[test]
    fn logspace_sub_very_close() {
        // Test when inputs are extremely close
        let a = 100.0;
        let b = 100.0 - 1e-12;
        let result = logspace_sub(a, b).unwrap();

        assert!(result.is_finite());
        assert!(result < a);
    }

    #[test]
    fn logspace_sub_equal() {
        // Test when a == b (should return NEG_INFINITY)
        let result = logspace_sub(5.0, 5.0).unwrap();
        assert_eq!(result, f64::NEG_INFINITY);
    }

    #[test]
    fn logspace_sub_invalid_order() {
        // Test when b > a (should return None)
        let result = logspace_sub(3.0, 5.0);
        assert!(result.is_none());
    }

    #[test]
    fn logspace_sub_infinite_inputs() {
        // Test with infinite inputs
        assert!(logspace_sub(f64::INFINITY, 5.0).is_none());
        // NEG_INFINITY is not finite, so should return None
        assert!(logspace_sub(5.0, f64::NEG_INFINITY).is_none());
        assert!(logspace_sub(f64::INFINITY, f64::INFINITY).is_none());
        assert!(logspace_sub(f64::NEG_INFINITY, f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn logspace_sub_nan_inputs() {
        // Test with NaN inputs
        assert!(logspace_sub(f64::NAN, 5.0).is_none());
        assert!(logspace_sub(5.0, f64::NAN).is_none());
    }
}
