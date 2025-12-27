use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

mod results;
mod scheduler;
mod state;

use super::SamplingResults;
use crate::common::{AskResult, Bounds, Point};
use crate::errors::EvaluationError;
use crate::optimisers::ScalarEvaluation;
use crate::prelude::TellError;
use crate::sampler::errors::SamplerTermination;
pub use results::{NestedSample, NestedSamples};

use state::LivePoint;

const MIN_LIVE_POINTS: usize = 8;
const MAX_ITERATION_MULTIPLIER: usize = 1024;
const INITIAL_EVAL_BATCH_SIZE: usize = 16;

#[derive(Debug, Clone)]
pub enum DNSPhase {
    InitialisingLivePoints {
        collected: Vec<LivePoint>,
        target: usize,
        attempts: usize,
    },
    AwaitingSingleReplacement {
        pending_position: Vec<f64>,
        removed: state::RemovedPoint,
        threshold: f64,
    },
    AwaitingExpansion {
        pending_positions: Vec<Vec<f64>>,
        target_live: usize,
    },
    Terminated(SamplerTermination),
}

pub struct DynamicNestedSamplerState {
    config: DynamicNestedSampler,
    bounds: Bounds,
    sampler_state: state::SamplerState,
    scheduler: scheduler::Scheduler,
    iterations: usize,
    max_iterations: usize,
    rng: StdRng,
    phase: DNSPhase,
    start_time: Instant,
}

/// Configurable Dynamic Nested Sampling engine
#[derive(Clone, Debug)]
pub struct DynamicNestedSampler {
    live_points: usize,
    expansion_factor: f64,
    termination_tol: f64,
    seed: Option<u64>,
}

/// Builder-style configuration and execution entry points
impl DynamicNestedSampler {
    /// Create a sampler with default live-point budget and tolerances.
    pub fn new() -> Self {
        Self {
            live_points: 64,
            expansion_factor: 0.5,
            termination_tol: 1e-3,
            seed: None,
        }
    }

    /// Set the number of live points, clamping to the algorithm minimum.
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

    /// Fix the seed for reproducible sampling runs.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initialise ask/tell state for external control of sampling
    ///
    /// Generates an initial batch of candidate points within bounds for evaluation.
    /// The initial parameter is currently ignored; initialisation samples from bounds.
    ///
    /// # Arguments
    /// * `initial` - Starting point (currently unused, reserved for future use)
    /// * `bounds` - Parameter bounds for sampling
    ///
    /// # Returns
    /// Tuple of (state, initial_candidates) where candidates should be evaluated
    pub fn init(&self, _initial: Point, bounds: Bounds) -> (DynamicNestedSamplerState, Vec<Point>) {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rand::rng()),
        };

        let dimension = bounds.dimension();

        // Generate first batch of candidates from bounds
        let target = self.live_points.max(MIN_LIVE_POINTS);
        let initial_batch_size = INITIAL_EVAL_BATCH_SIZE.min(target);

        let mut candidates = Vec::with_capacity(initial_batch_size);
        for _ in 0..initial_batch_size {
            let mut position = bounds.sample(&mut rng, self.expansion_factor);
            bounds.clamp(&mut position);
            candidates.push(position);
        }

        let max_iterations = MAX_ITERATION_MULTIPLIER
            .saturating_mul(self.live_points)
            .saturating_mul(dimension.max(1));

        let state = DynamicNestedSamplerState {
            config: self.clone(),
            bounds,
            sampler_state: state::SamplerState::new(Vec::new()),
            scheduler: scheduler::Scheduler::new(
                self.live_points,
                self.expansion_factor,
                self.termination_tol,
            ),
            iterations: 0,
            max_iterations,
            rng,
            phase: DNSPhase::InitialisingLivePoints {
                collected: Vec::new(),
                target,
                attempts: 0,
            },
            start_time: Instant::now(),
        };

        (state, candidates)
    }
}

impl DynamicNestedSamplerState {
    /// Get next point(s) to evaluate
    ///
    /// Returns either:
    /// - `Evaluate(points)`: Batch of points to evaluate, then call `tell()` with results
    /// - `Done(results)`: Sampling complete, contains final `NestedSamples`
    pub fn ask(&self) -> AskResult<SamplingResults> {
        match &self.phase {
            DNSPhase::Terminated(reason) => {
                AskResult::Done(SamplingResults::Nested(self.build_results()))
            }
            DNSPhase::InitialisingLivePoints {
                collected,
                target,
                attempts,
            } => {
                // Check if we have enough points or exceeded max attempts
                if collected.len() >= *target || *attempts >= target.saturating_mul(200).max(1000) {
                    // Should transition to main loop, but this shouldn't happen
                    // because tell() handles the transition
                    return AskResult::Done(SamplingResults::Nested(self.build_results()));
                }

                // Request next batch
                let remaining = target.saturating_sub(collected.len());
                let batch_size = INITIAL_EVAL_BATCH_SIZE.min(remaining);

                let mut candidates = Vec::with_capacity(batch_size);
                let mut temp_rng = self.rng.clone();
                for _ in 0..batch_size {
                    let mut position = self
                        .bounds
                        .sample(&mut temp_rng, self.config.expansion_factor);
                    self.bounds.clamp(&mut position);
                    candidates.push(position);
                }

                AskResult::Evaluate(candidates)
            }
            DNSPhase::AwaitingSingleReplacement {
                pending_position, ..
            } => AskResult::Evaluate(vec![pending_position.clone()]),
            DNSPhase::AwaitingExpansion {
                pending_positions, ..
            } => AskResult::Evaluate(pending_positions.clone()),
        }
    }

    /// Report evaluation results for proposed points
    ///
    /// Process the results and advance the sampling state machine.
    /// Errors are treated as infinite values (rejected).
    ///
    /// # Arguments
    /// * `results` - Evaluation results matching the last ask() request
    ///
    /// # Returns
    /// `Ok(())` on success, or `TellError` if already terminated or result count mismatch
    pub fn tell<I, T, E>(&mut self, results: I) -> Result<(), TellError>
    where
        I: IntoIterator<Item = T>,
        T: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        // Check for termination
        if matches!(self.phase, DNSPhase::Terminated(_)) {
            return Err(TellError::AlreadyTerminated);
        }

        // Check iteration limit
        if self.iterations >= self.max_iterations {
            self.phase = DNSPhase::Terminated(SamplerTermination::MaxIterationReached);
            return Ok(());
        }

        // Convert results to Vec<f64>, treating errors as INFINITY
        let values: Vec<f64> = results
            .into_iter()
            .map(|r| match r.try_into() {
                Ok(eval) => eval.value(),
                Err(_) => f64::INFINITY,
            })
            .collect();

        // Dispatch to phase handler
        match &self.phase {
            DNSPhase::InitialisingLivePoints { .. } => self.handle_initialisation(values),
            DNSPhase::AwaitingSingleReplacement { .. } => self.handle_single_replacement(values),
            DNSPhase::AwaitingExpansion { .. } => self.handle_expansion(values),
            DNSPhase::Terminated(_) => unreachable!("Already checked above"),
        }
    }

    /// Handle initialization phase results
    fn handle_initialisation(&mut self, values: Vec<f64>) -> Result<(), TellError> {
        let (collected, target, attempts) = match &mut self.phase {
            DNSPhase::InitialisingLivePoints {
                collected,
                target,
                attempts,
            } => (collected, *target, attempts),
            _ => unreachable!(),
        };

        // Take phase to avoid borrow issues
        let mut temp_collected = std::mem::take(collected);
        let mut temp_attempts = *attempts;

        // Get the candidates that were evaluated (reconstruct from RNG state)
        let batch_size = values.len();
        let mut candidates = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut position = self
                .bounds
                .sample(&mut self.rng, self.config.expansion_factor);
            self.bounds.clamp(&mut position);
            candidates.push(position);
        }

        // Process results
        for (position, value) in candidates.into_iter().zip(values) {
            temp_attempts += 1;
            let log_likelihood = -value;

            if log_likelihood.is_finite() {
                temp_collected.push(LivePoint::new(position, log_likelihood));
            }

            if temp_collected.len() >= target {
                break;
            }
        }

        // Check if initialization is complete
        if temp_collected.len() >= target {
            // Move to main sampling loop
            self.sampler_state = state::SamplerState::new(temp_collected);
            self.start_next_iteration()
        } else if temp_attempts >= target.saturating_mul(200).max(1000) {
            // Failed to initialize - terminate
            self.phase = DNSPhase::Terminated(SamplerTermination::InsufficientLivePoints);
            Ok(())
        } else {
            // Continue initialization
            self.phase = DNSPhase::InitialisingLivePoints {
                collected: temp_collected,
                target,
                attempts: temp_attempts,
            };
            Ok(())
        }
    }

    /// Handle single replacement phase results
    fn handle_single_replacement(&mut self, mut values: Vec<f64>) -> Result<(), TellError> {
        if values.len() != 1 {
            return Err(TellError::ResultCountMismatch {
                expected: 1,
                got: values.len(),
            });
        }

        let value = values.remove(0);
        let (position, removed, threshold) = match &self.phase {
            DNSPhase::AwaitingSingleReplacement {
                pending_position,
                removed,
                threshold,
            } => (pending_position.clone(), removed.clone(), *threshold),
            _ => unreachable!(),
        };

        let log_likelihood = -value;

        if log_likelihood.is_finite() && log_likelihood > threshold {
            // Accept the new point
            self.sampler_state.accept_removed(removed);
            self.sampler_state
                .insert_live_point(LivePoint::new(position, log_likelihood));
        } else {
            // Reject - restore the removed point
            self.sampler_state.restore_removed(removed);
        }

        self.iterations += 1;
        self.start_next_iteration()
    }

    /// Handle expansion phase results
    fn handle_expansion(&mut self, values: Vec<f64>) -> Result<(), TellError> {
        let (positions, target_live) = match &self.phase {
            DNSPhase::AwaitingExpansion {
                pending_positions,
                target_live,
            } => (pending_positions.clone(), *target_live),
            _ => unreachable!(),
        };

        if values.len() != positions.len() {
            return Err(TellError::ResultCountMismatch {
                expected: positions.len(),
                got: values.len(),
            });
        }

        // Add all valid points to live set
        for (position, value) in positions.into_iter().zip(values) {
            let log_likelihood = -value;
            if log_likelihood.is_finite() {
                self.sampler_state
                    .insert_live_point(LivePoint::new(position, log_likelihood));
            }

            if self.sampler_state.live_point_count() >= target_live {
                break;
            }
        }

        self.start_next_iteration()
    }

    /// Start next iteration of the main sampling loop
    fn start_next_iteration(&mut self) -> Result<(), TellError> {
        // Check termination conditions
        if self.sampler_state.live_points().is_empty() {
            self.phase = DNSPhase::Terminated(SamplerTermination::InsufficientLivePoints);
            return Ok(());
        }

        if self.iterations >= self.max_iterations {
            self.phase = DNSPhase::Terminated(SamplerTermination::MaxIterationReached);
            return Ok(());
        }

        // Update scheduler and check for termination
        let info_estimate = results::information_estimate(self.sampler_state.posterior());
        let current_live = self.sampler_state.live_point_count();
        let target_live = self.scheduler.target(info_estimate, current_live);

        if self
            .scheduler
            .should_terminate(&self.sampler_state, info_estimate)
        {
            self.sampler_state.finalize();
            self.phase = DNSPhase::Terminated(SamplerTermination::EvidenceConverged);
            return Ok(());
        }

        // Adjust live set if needed
        self.sampler_state.adjust_live_set(target_live);

        // Check if we need to expand the live set
        if self.sampler_state.live_point_count() < target_live {
            let needed = target_live.saturating_sub(self.sampler_state.live_point_count());
            let batch_size = needed.min(16); // Limit batch size

            let mut positions = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let threshold = self.sampler_state.min_log_likelihood();
                let position = self.generate_proposal(threshold);
                positions.push(position);
            }

            self.phase = DNSPhase::AwaitingExpansion {
                pending_positions: positions,
                target_live,
            };
            return Ok(());
        }

        // Remove worst point and request replacement
        if let Some(worst_index) = self.sampler_state.worst_index() {
            if let Some(removed) = self.sampler_state.remove_at(worst_index) {
                let threshold = removed.log_likelihood();
                let position = self.generate_proposal(threshold);

                self.phase = DNSPhase::AwaitingSingleReplacement {
                    pending_position: position,
                    removed,
                    threshold,
                };
                return Ok(());
            }
        }

        // No worst point - terminate
        self.sampler_state.finalize();
        self.phase = DNSPhase::Terminated(SamplerTermination::InsufficientLivePoints);
        Ok(())
    }

    /// Generate a proposal point using the proposal engine
    fn generate_proposal(&mut self, threshold: f64) -> Vec<f64> {
        let live_points = self.sampler_state.live_points();

        if live_points.is_empty() {
            let mut position = self
                .bounds
                .sample(&mut self.rng, self.config.expansion_factor);
            self.bounds.clamp(&mut position);
            return position;
        }

        // Use proposal engine's logic without evaluation
        let step_sizes =
            self.compute_scales(live_points, &self.bounds, self.config.expansion_factor);
        let max_attempts = 64;

        for attempt in 0..max_attempts {
            let candidate = if attempt % 32 == 0 {
                let mut position = self
                    .bounds
                    .sample(&mut self.rng, self.config.expansion_factor);
                self.bounds.clamp(&mut position);
                position
            } else {
                use rand::Rng;
                let anchor_idx = self.rng.random_range(0..live_points.len());
                let anchor = &live_points[anchor_idx];
                let mut proposal = anchor.position.clone();

                for (value, scale) in proposal.iter_mut().zip(step_sizes.iter()) {
                    use rand_distr::StandardNormal;
                    let perturb: f64 = self.rng.sample(StandardNormal);
                    *value += perturb * scale;
                }

                self.bounds.clamp(&mut proposal);
                proposal
            };

            // Return first candidate (evaluation will happen in ask/tell cycle)
            return candidate;
        }

        // Fallback: sample from bounds
        let mut position = self
            .bounds
            .sample(&mut self.rng, self.config.expansion_factor);
        self.bounds.clamp(&mut position);
        position
    }

    /// Compute Gaussian perturbation scales per dimension
    fn compute_scales(
        &self,
        live_points: &[LivePoint],
        bounds: &Bounds,
        expansion_factor: f64,
    ) -> Vec<f64> {
        let dimension = bounds.dimension();
        if live_points.is_empty() {
            return vec![expansion_factor.max(0.1); dimension];
        }

        let mut mins = vec![f64::INFINITY; dimension];
        let mut maxs = vec![f64::NEG_INFINITY; dimension];

        for point in live_points {
            for (i, value) in point.position.iter().enumerate() {
                mins[i] = mins[i].min(*value);
                maxs[i] = maxs[i].max(*value);
            }
        }

        mins.iter_mut().zip(maxs.iter_mut()).for_each(|(min, max)| {
            if !min.is_finite() {
                *min = -1.0;
            }
            if !max.is_finite() {
                *max = 1.0;
            }
            if *max <= *min {
                *max = *min + 1e-3;
            }
        });

        mins.iter()
            .zip(maxs.iter())
            .map(|(&min, &max)| {
                let width = (max - min).abs().max(1e-3);
                width * expansion_factor.max(0.05)
            })
            .collect()
    }

    /// Build final results from current state
    fn build_results(&self) -> NestedSamples {
        let mut result = results::NestedSamples::build(
            self.sampler_state.posterior(),
            self.sampler_state.dimension(),
        );
        result.set_time(self.start_time.elapsed());
        result
    }

    /// Query methods for state inspection
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn live_point_count(&self) -> usize {
        self.sampler_state.live_point_count()
    }
}

impl DynamicNestedSampler {
    /// Run Dynamic Nested Sampling with automatic evaluation loop
    ///
    /// Internally uses the ask/tell interface. For external control,
    /// use `init()`, `ask()`, and `tell()` directly.
    ///
    /// # Arguments
    /// * `objective` - Function to evaluate (returns negative log-likelihood)
    /// * `initial` - Initial point (currently unused, reserved for future)
    /// * `bounds` - Parameter bounds
    pub fn run<F, R, E>(&self, mut objective: F, initial: Point, bounds: Bounds) -> NestedSamples
    where
        F: FnMut(&[f64]) -> R,
        R: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        let (mut state, first_batch) = self.init(initial, bounds);
        let mut results: Vec<_> = first_batch.iter().map(|p| objective(p)).collect();

        loop {
            state.tell(results).ok();
            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| objective(p)).collect();
                }
                AskResult::Done(SamplingResults::Nested(samples)) => return samples,
                _ => unreachable!("DynamicNestedSampler always returns Nested results"),
            }
        }
    }
}

impl Default for DynamicNestedSampler {
    fn default() -> Self {
        Self::new()
    }
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
    use crate::builders::ScalarProblemBuilder;
    use crate::problem::ParameterSpec;
    fn gaussian_problem(
        mean: f64,
        sigma: f64,
    ) -> crate::problem::Problem<impl crate::problem::Objective> {
        let log_norm = sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
        ScalarProblemBuilder::new()
            .with_function(move |x: &[f64]| {
                let diff = x[0] - mean;
                0.5 * (diff * diff) / (sigma * sigma) + log_norm
            })
            .with_parameter("x", mean, (mean - 10.0, mean + 10.0))
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

        let nested = sampler.run(|x| problem.evaluate(x), vec![1.5], Bounds::unbounded(1));

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
