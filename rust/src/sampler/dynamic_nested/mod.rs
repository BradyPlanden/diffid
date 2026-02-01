//! Dynamic Nested Sampler for Bayesian evidence estimation.
//!
//! # Objective Function Convention
//!
//! **IMPORTANT**: The objective function must return **negative log-likelihood** (-log L).
//! Internally, values are negated to obtain log-likelihood for nested sampling calculations.
//!
//! This convention aligns with optimisation where lower values are better, while nested
//! sampling requires higher log-likelihood values.

use rand::prelude::StdRng;
use rand::SeedableRng;
use std::time::{Duration, Instant};

mod mcmc_proposal;
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
        pending_positions: Vec<Vec<f64>>,
        target: usize,
        attempts: usize,
    },
    AwaitingReplacementBatch {
        pending_positions: Vec<Vec<f64>>,
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
    mcmc_generator: mcmc_proposal::MCMCProposalGenerator,
}

/// Configurable Dynamic Nested Sampling engine
#[must_use]
#[derive(Clone, Debug)]
pub struct DynamicNestedSampler {
    live_points: usize,
    expansion_factor: f64,
    termination_tol: f64,
    seed: Option<u64>,
    mcmc_batch_size: usize,
    mcmc_step_size: f64,
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
            mcmc_batch_size: 8,
            mcmc_step_size: 0.01,
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

    /// Set the MCMC batch size for proposal generation.
    ///
    /// When replacing live points, this many proposals are generated via MCMC
    /// and evaluated in parallel. The best valid proposal (likelihood > threshold)
    /// is selected. Larger batches increase the chance of finding valid proposals
    /// but require more evaluations per iteration.
    ///
    /// Default: 8
    pub fn with_mcmc_batch_size(mut self, size: usize) -> Self {
        self.mcmc_batch_size = size.max(1);
        self
    }

    /// Set the MCMC step size factor for proposal generation.
    ///
    /// Controls the size of Gaussian perturbations relative to bounds width.
    /// Step size in dimension i is: `mcmc_step_size × (upper[i] - lower[i])`
    ///
    /// Smaller values (0.001-0.005): Conservative, local exploration
    /// Medium values (0.01-0.02): Balanced (default: 0.01)
    /// Larger values (0.03-0.1): Aggressive, global exploration
    ///
    /// Default: 0.01
    pub fn with_mcmc_step_size(mut self, step_size: f64) -> Self {
        self.mcmc_step_size = step_size.max(1e-6);
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
    /// Tuple of (state, `initial_candidates`) where candidates should be evaluated
    pub fn init(&self, _initial: Point, bounds: Bounds) -> (DynamicNestedSamplerState, Vec<Point>) {
        let mut rng = self
            .seed
            .map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64);

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
                pending_positions: candidates.clone(),
                target,
                attempts: 0,
            },
            start_time: Instant::now(),
            mcmc_generator: mcmc_proposal::MCMCProposalGenerator::new(self.mcmc_step_size),
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
            DNSPhase::Terminated(_reason) => {
                AskResult::Done(SamplingResults::Nested(self.build_results()))
            }
            DNSPhase::InitialisingLivePoints {
                collected,
                pending_positions,
                target,
                attempts,
            } => {
                // Check if we have enough points or exceeded max attempts
                if collected.len() >= *target || *attempts >= target.saturating_mul(200).max(1000) {
                    // Should transition to main loop, but this shouldn't happen
                    // because tell() handles the transition
                    return AskResult::Done(SamplingResults::Nested(self.build_results()));
                }

                // Return stored pending positions (like other phases)
                AskResult::Evaluate(pending_positions.clone())
            }
            DNSPhase::AwaitingReplacementBatch {
                pending_positions, ..
            }
            | DNSPhase::AwaitingExpansion {
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
    /// * `results` - Evaluation results matching the last `ask()` request
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The sampler has already terminated
    /// - The number of results doesn't match the expected count
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
            .map(|r| r.try_into().map_or(f64::INFINITY, |eval| eval.value()))
            .collect();

        // Dispatch to phase handler
        match &self.phase {
            DNSPhase::InitialisingLivePoints { .. } => self.handle_initialisation(values),
            DNSPhase::AwaitingReplacementBatch { .. } => self.handle_replacement_batch(&values),
            DNSPhase::AwaitingExpansion { .. } => self.handle_expansion(values),
            DNSPhase::Terminated(_) => unreachable!("Already checked above"),
        }
    }

    /// Handle initialization phase results
    fn handle_initialisation(&mut self, values: Vec<f64>) -> Result<(), TellError> {
        let (collected, pending_positions, target, attempts) = match &mut self.phase {
            DNSPhase::InitialisingLivePoints {
                collected,
                pending_positions,
                target,
                attempts,
            } => (collected, pending_positions, *target, attempts),
            _ => unreachable!(),
        };

        // Take ownership of phase
        let mut temp_collected = std::mem::take(collected);
        let temp_pending = std::mem::take(pending_positions);
        let mut temp_attempts = *attempts;

        // Validate result count matches pending positions
        if values.len() != temp_pending.len() {
            return Err(TellError::ResultCountMismatch {
                expected: temp_pending.len(),
                got: values.len(),
            });
        }

        // Process results using stored positions
        for (position, value) in temp_pending.into_iter().zip(values) {
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
            // Continue initialization - generate new batch for next iteration
            let remaining = target.saturating_sub(temp_collected.len());
            let batch_size = INITIAL_EVAL_BATCH_SIZE.min(remaining);

            let mut new_pending = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let mut position = self
                    .bounds
                    .sample(&mut self.rng, self.config.expansion_factor);
                self.bounds.clamp(&mut position);
                new_pending.push(position);
            }

            self.phase = DNSPhase::InitialisingLivePoints {
                collected: temp_collected,
                pending_positions: new_pending,
                target,
                attempts: temp_attempts,
            };
            Ok(())
        }
    }

    /// Handle replacement batch phase results
    ///
    /// Evaluates a batch of MCMC proposals and selects the best valid one (likelihood > threshold).
    /// If all proposals are rejected, the removed point is restored.
    fn handle_replacement_batch(&mut self, values: &[f64]) -> Result<(), TellError> {
        // Extract phase data
        let (positions, removed, threshold) = match &self.phase {
            DNSPhase::AwaitingReplacementBatch {
                pending_positions,
                removed,
                threshold,
            } => (pending_positions.clone(), removed.clone(), *threshold),
            _ => unreachable!(),
        };

        // Validate result count
        if values.len() != positions.len() {
            return Err(TellError::ResultCountMismatch {
                expected: positions.len(),
                got: values.len(),
            });
        }

        // Find valid proposals (likelihood > threshold)
        let mut valid_proposals = Vec::new();
        for (position, value) in positions.iter().zip(values.iter()) {
            let log_likelihood = -value;
            if log_likelihood.is_finite() && log_likelihood > threshold {
                valid_proposals.push((position.clone(), log_likelihood));
            }
        }

        // Accept the best valid proposal or restore removed point
        if let Some((best_position, best_likelihood)) = valid_proposals
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            // Accept: finalize removal and insert new point
            self.sampler_state.accept_removed(removed);
            self.sampler_state
                .insert_live_point(LivePoint::new(best_position, best_likelihood));
        } else {
            // All rejected: restore removed point (NO LOOP!)
            self.sampler_state.restore_removed(removed);
        }

        // Continue iteration
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
    #[allow(clippy::unnecessary_wraps)]
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

            // For expansion, use uniform sampling from bounds (simpler for initial population)
            let mut positions = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let mut position = self
                    .bounds
                    .sample(&mut self.rng, self.config.expansion_factor);
                self.bounds.clamp(&mut position);
                positions.push(position);
            }

            self.phase = DNSPhase::AwaitingExpansion {
                pending_positions: positions,
                target_live,
            };
            return Ok(());
        }

        // Remove worst point and generate MCMC replacement batch
        if let Some(worst_index) = self.sampler_state.worst_index() {
            if let Some(removed) = self.sampler_state.remove_at(worst_index) {
                let threshold = removed.log_likelihood();

                // Guard against empty live points after removal
                if self.sampler_state.live_point_count() == 0 {
                    self.sampler_state.restore_removed(removed);
                    self.phase = DNSPhase::Terminated(SamplerTermination::InsufficientLivePoints);
                    return Ok(());
                }

                // Select random live point as MCMC starting point
                let start_point = self.sampler_state.random_live_point(&mut self.rng);

                // Generate batch of MCMC proposals
                let pending_positions = self.mcmc_generator.generate_batch(
                    start_point.position(),
                    &self.bounds,
                    &mut self.rng,
                    self.config.mcmc_batch_size,
                );

                self.phase = DNSPhase::AwaitingReplacementBatch {
                    pending_positions,
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

    pub fn phase(&self) -> &DNSPhase {
        &self.phase
    }
}

impl DynamicNestedSampler {
    /// Run Dynamic Nested Sampling with automatic evaluation loop
    ///
    /// Internally uses the ask/tell interface. For external control,
    /// use `init()`, `ask()`, and `tell()` directly.
    ///
    /// # Arguments
    /// * `objective` - Function to evaluate. **Must return negative log-likelihood** (-log L).
    ///   The sampler will negate this internally to obtain log-likelihood for
    ///   nested sampling calculations.
    /// * `initial` - Initial point (currently unused, reserved for future)
    /// * `bounds` - Parameter bounds
    ///
    /// # Panics
    ///
    /// Panics if `tell()` fails during the sampling loop, which indicates a bug in the sampler.
    pub fn run<F, R, E>(&self, mut objective: F, initial: Point, bounds: Bounds) -> NestedSamples
    where
        F: FnMut(&[f64]) -> R,
        R: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        let (mut state, first_batch) = self.init(initial, bounds);
        let mut results: Vec<_> = first_batch.iter().map(|p| objective(p)).collect();

        loop {
            state
                .tell(results)
                .expect("tell() failed - this indicates a bug in the sampler");
            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| objective(p)).collect();
                }
                AskResult::Done(SamplingResults::Nested(samples)) => return samples,
                AskResult::Done(_) => {
                    unreachable!("DynamicNestedSampler always returns Nested results")
                }
            }
        }
    }

    /// Run Dynamic Nested Sampling with automatic evaluation loop
    ///
    /// Internally uses the ask/tell interface. For external control,
    /// use `init()`, `ask()`, and `tell()` directly.
    ///
    /// # Arguments
    /// * `objective` - Function to evaluate. **Must return negative log-likelihood** (-log L).
    ///   The sampler will negate this internally to obtain log-likelihood for
    ///   nested sampling calculations.
    /// * `initial` - Initial point (currently unused, reserved for future)
    /// * `bounds` - Parameter bounds
    ///
    /// # Panics
    ///
    /// Panics if the sampling state machine enters an unexpected state after a tell error.
    /// This should not occur under normal operation.
    pub fn run_batch<F, R, E>(&self, objective: F, initial: Point, bounds: Bounds) -> NestedSamples
    where
        F: Fn(&[Vec<f64>]) -> Vec<R>,
        R: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        let (mut state, first_batch) = self.init(initial, bounds);
        let mut results = objective(&first_batch);

        loop {
            // Call ask and break if an error is encountered
            if state.tell(results).is_err() {
                break;
            }

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = objective(&points);
                }
                AskResult::Done(SamplingResults::Nested(samples)) => return samples,
                AskResult::Done(_) => {
                    unreachable!("DynamicNestedSampler always returns Nested results")
                }
            }
        }

        // Final ask call for if tell returned an error
        match state.ask() {
            AskResult::Done(SamplingResults::Nested(samples)) => samples,
            _ => panic!("Unexpected state after tell error"),
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

    // Use relative tolerance for numerical stability
    let rel_tol = 1e-15 * a.abs().max(b.abs()).max(1.0);
    if (a - b).abs() < rel_tol {
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
            "posterior mean must be finite, got {mean:.4}"
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
            .map(super::results::NestedSample::evidence_weight)
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

    // DNS-Specific Ask/Tell Tests
    #[test]
    fn dns_init_returns_correct_batch_size() {
        let sampler = DynamicNestedSampler::new()
            .with_live_points(64)
            .with_seed(42);

        let (state, initial_batch) = sampler.init(vec![0.0], Bounds::unbounded(1));

        // First batch should be min(INITIAL_EVAL_BATCH_SIZE, target_live_points)
        let expected_batch = INITIAL_EVAL_BATCH_SIZE.min(64);
        assert_eq!(
            initial_batch.len(),
            expected_batch,
            "Initial batch size should be min(INITIAL_EVAL_BATCH_SIZE, live_points)"
        );

        // All points should be within bounds
        for point in &initial_batch {
            assert_eq!(point.len(), 1, "Points should be 1-dimensional");
            assert!(point[0].is_finite(), "Points should be finite");
        }

        // State should start in InitialisingLivePoints
        assert!(
            matches!(state.phase(), DNSPhase::InitialisingLivePoints { .. }),
            "Should start in InitialisingLivePoints phase"
        );
    }

    #[test]
    fn dns_tell_initialization_collects_points() {
        let sampler = DynamicNestedSampler::new()
            .with_live_points(32)
            .with_seed(42);

        let (mut state, initial_batch) = sampler.init(vec![0.0], Bounds::unbounded(1));

        // Provide results for initial batch
        let results: Vec<_> = initial_batch
            .iter()
            .map(|x| Ok::<f64, std::io::Error>(0.5 * x[0].powi(2)))
            .collect();
        state.tell(results).unwrap();

        // After first tell, should still be initializing or have transitioned
        match state.phase() {
            DNSPhase::InitialisingLivePoints {
                collected, target, ..
            } => {
                assert!(
                    collected.len() <= *target,
                    "Collected should not exceed target"
                );
            }
            DNSPhase::AwaitingReplacementBatch { .. }
            | DNSPhase::AwaitingExpansion { .. }
            | DNSPhase::Terminated(_) => {
                // Transitioned out of initialization - ok
            }
        }
    }

    #[test]
    fn dns_expansion_phase_batching() {
        let sampler = DynamicNestedSampler::new()
            .with_live_points(32)
            .with_seed(42);

        let (mut state, initial_batch) = sampler.init(vec![0.0], Bounds::unbounded(1));

        // Provide initial results
        let results: Vec<_> = initial_batch
            .iter()
            .map(|x| Ok::<f64, std::io::Error>(0.5 * x[0].powi(2)))
            .collect();
        state.tell(results).unwrap();

        // Run until we potentially see an expansion phase
        let mut seen_expansion = false;
        let mut max_batch_size = 0;

        for _ in 0..100 {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    max_batch_size = max_batch_size.max(points.len());

                    // Check if we're in expansion phase (multiple points)
                    if let DNSPhase::AwaitingExpansion { .. } = state.phase() {
                        seen_expansion = true;
                        // Expansion batches should be limited
                        assert!(
                            points.len() <= 16,
                            "Expansion batch should be limited to max 16"
                        );
                    }

                    let results: Vec<_> = points
                        .iter()
                        .map(|x| Ok::<f64, std::io::Error>(0.5 * x[0].powi(2)))
                        .collect();
                    state.tell(results).unwrap();
                }
                AskResult::Done(_) => break,
            }
        }

        // We should see some batching (either expansion or initialization)
        assert!(
            max_batch_size > 1 || seen_expansion,
            "Should see batched evaluations or expansion phase"
        );
    }

    #[test]
    fn dns_max_iteration_termination() {
        // Create a sampler with small live point count
        let sampler = DynamicNestedSampler::new()
            .with_live_points(16) // Small live point count
            .with_seed(42);

        let (mut state, initial_batch) = sampler.init(vec![0.0], Bounds::unbounded(1));

        // Provide initial results with proper objective function
        let results: Vec<_> = initial_batch
            .iter()
            .map(|x| Ok::<f64, std::io::Error>(0.5 * x[0].powi(2)))
            .collect();
        state.tell(results).unwrap();

        let mut iterations = 0;
        let max_allowed = 100_000; // Safety limit to prevent infinite loop

        loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    let results: Vec<_> = points
                        .iter()
                        .map(|x| Ok::<f64, std::io::Error>(0.5 * x[0].powi(2)))
                        .collect();
                    state.tell(results).unwrap();
                    iterations += 1;

                    assert!(
                        iterations <= max_allowed,
                        "Exceeded safety limit - sampler not terminating"
                    );
                }
                AskResult::Done(SamplingResults::Nested(samples)) => {
                    // Should terminate eventually
                    assert!(iterations > 0, "Should have run some iterations");
                    // Sampler should complete and return results (draws may be small with few live points)
                    let _draws = samples.draws(); // Verify we can get draws

                    // Check termination reason from phase
                    if let DNSPhase::Terminated(reason) = state.phase() {
                        match reason {
                            SamplerTermination::MaxIterationReached
                            | SamplerTermination::EvidenceConverged
                            | SamplerTermination::InformationConverged
                            | SamplerTermination::InsufficientLivePoints => {
                                // Expected termination reasons
                            }
                            other @ SamplerTermination::EvaluationFailed(_) => {
                                // Other reasons are also ok, just document what we see
                                eprintln!("Terminated with reason: {other:?}");
                            }
                        }
                    }
                    break;
                }
                AskResult::Done(_) => panic!("Unexpected result"),
            }
        }
    }
}
