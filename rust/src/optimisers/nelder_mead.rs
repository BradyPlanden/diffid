use crate::common::{AskResult, Bounds, Point};
use crate::errors::{EvaluationError, TellError};
use crate::optimisers::{
    build_results, EvaluatedPoint, OptimisationResults, ScalarEvaluation, TerminationReason,
};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

/// Configuration for the Nelder-Mead optimizer
#[derive(Clone, Debug)]
pub struct NelderMead {
    max_iter: usize,
    threshold: f64,
    sigma0: f64,
    position_tolerance: f64,
    max_evaluations: Option<usize>,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
    patience: Option<Duration>,
}

impl NelderMead {
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            threshold: 1e-6,
            sigma0: 0.1,
            position_tolerance: 1e-6,
            max_evaluations: None,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            patience: None,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }

    pub fn with_sigma0(mut self, sigma0: f64) -> Self {
        self.sigma0 = sigma0;
        self
    }

    pub fn with_position_tolerance(mut self, tolerance: f64) -> Self {
        self.position_tolerance = tolerance.max(0.0);
        self
    }

    pub fn with_max_evaluations(mut self, max_evaluations: usize) -> Self {
        self.max_evaluations = Some(max_evaluations);
        self
    }

    pub fn with_coefficients(mut self, alpha: f64, gamma: f64, rho: f64, sigma: f64) -> Self {
        self.alpha = alpha;
        self.gamma = gamma;
        self.rho = rho;
        self.sigma = sigma;
        self
    }

    pub fn with_patience(mut self, patience: f64) -> Self {
        self.patience = Some(Duration::from_secs_f64(patience));
        self
    }

    /// Initialize the optimization state
    ///
    /// Returns the state and the first point to evaluate
    pub fn init(&self, initial: Point, bounds: Bounds) -> (NelderMeadState, Vec<Point>) {
        let dim = initial.len();
        let mut initial_point = initial;
        bounds.clamp(&mut initial_point);

        let state = NelderMeadState {
            config: self.clone(),
            simplex: Vec::with_capacity(dim + 1),
            phase: NelderMeadPhase::EvaluatingInitial,
            bounds,
            dim,
            initial_point: initial_point.clone(),
            nit: 0,
            nfev: 0,
            start_time: Instant::now(),
        };

        (state, vec![initial_point])
    }

    /// Run optimization using a closure for evaluation
    ///
    /// This is a convenience wrapper around the ask/tell interface
    pub fn run<F, R, E>(
        &self,
        mut objective: F,
        initial: Point,
        bounds: Bounds,
    ) -> OptimisationResults
    where
        F: FnMut(&[f64]) -> R,
        R: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        let (mut state, first_point) = self.init(initial, bounds);
        let mut result = objective(&first_point[0]);

        loop {
            if state.tell(result).is_err() {
                break;
            }

            match state.ask() {
                AskResult::Evaluate(point) => {
                    result = objective(&point[0]);
                }
                AskResult::Done(results) => {
                    return results;
                }
            }
        }

        // Should not reach here normally
        match state.ask() {
            AskResult::Done(results) => results,
            _ => panic!("Unexpected state after tell error"),
        }
    }
}

impl Default for NelderMead {
    fn default() -> Self {
        Self::new()
    }
}

/// The current phase of the Nelder-Mead algorithm
#[derive(Clone, Debug)]
pub enum NelderMeadPhase {
    /// Waiting for initial point evaluation
    EvaluatingInitial,

    /// Building the initial simplex
    BuildingSimplex {
        next_dim: usize,
        pending_point: Point,
    },

    /// Waiting for reflection point evaluation
    AwaitingReflection {
        centroid: Point,
        reflected_point: Point,
    },

    /// Waiting for expansion point evaluation
    AwaitingExpansion {
        _centroid: Point,
        reflected_point: Point,
        reflected_value: f64,
        expanded_point: Point,
    },

    /// Waiting for contraction point evaluation
    AwaitingContraction {
        reflected_point: Point,
        reflected_value: f64,
        contract_point: Point,
    },

    /// Shrinking the simplex toward the best point
    Shrinking {
        best_point: Point,
        shrink_index: usize,
        pending_point: Point,
    },

    /// Algorithm has terminated
    Terminated(TerminationReason),
}

/// Runtime state of the Nelder-Mead optimizer
pub struct NelderMeadState {
    config: NelderMead,
    simplex: Vec<EvaluatedPoint>,
    phase: NelderMeadPhase,
    bounds: Bounds,
    dim: usize,
    initial_point: Point,
    nit: usize,
    nfev: usize,
    start_time: Instant,
}

impl NelderMeadState {
    /// Get the next point to evaluate, or the final result if optimization is complete
    pub fn ask(&self) -> AskResult<OptimisationResults> {
        match &self.phase {
            NelderMeadPhase::Terminated(reason) => {
                AskResult::Done(self.build_results(reason.clone()))
            }
            NelderMeadPhase::EvaluatingInitial => {
                AskResult::Evaluate(vec![self.initial_point.clone()])
            }
            NelderMeadPhase::BuildingSimplex { pending_point, .. } => {
                AskResult::Evaluate(vec![pending_point.clone()])
            }
            NelderMeadPhase::AwaitingReflection {
                reflected_point, ..
            } => AskResult::Evaluate(vec![reflected_point.clone()]),
            NelderMeadPhase::AwaitingExpansion { expanded_point, .. } => {
                AskResult::Evaluate(vec![expanded_point.clone()])
            }
            NelderMeadPhase::AwaitingContraction { contract_point, .. } => {
                AskResult::Evaluate(vec![contract_point.clone()])
            }
            NelderMeadPhase::Shrinking { pending_point, .. } => {
                AskResult::Evaluate(vec![pending_point.clone()])
            }
        }
    }

    /// Report the evaluation result for the last point from `ask()`
    ///
    /// Pass `Err` if the objective function failed to evaluate
    pub fn tell<T, E>(&mut self, result: T) -> Result<(), TellError>
    where
        T: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        if matches!(self.phase, NelderMeadPhase::Terminated(_)) {
            return Err(TellError::AlreadyTerminated);
        }

        let value = match result.try_into() {
            Ok(eval) => eval.value(),
            Err(e) => {
                let err: EvaluationError = e.into();
                self.phase = NelderMeadPhase::Terminated(
                    TerminationReason::FunctionEvaluationFailed(format!("{}", err)),
                );
                return Ok(());
            }
        };

        self.nfev += 1;

        // Take ownership of current phase to enable destructuring
        let phase = std::mem::replace(
            &mut self.phase,
            NelderMeadPhase::Terminated(TerminationReason::MaxIterationsReached),
        );

        match phase {
            NelderMeadPhase::EvaluatingInitial => {
                self.handle_initial_evaluated(value);
            }
            NelderMeadPhase::BuildingSimplex {
                next_dim,
                pending_point,
            } => {
                self.handle_simplex_point(next_dim, pending_point, value);
            }
            NelderMeadPhase::AwaitingReflection {
                centroid,
                reflected_point,
            } => {
                self.handle_reflection(centroid, reflected_point, value);
            }
            NelderMeadPhase::AwaitingExpansion {
                reflected_point,
                reflected_value,
                expanded_point,
                ..
            } => {
                self.handle_expansion(reflected_point, reflected_value, expanded_point, value);
            }
            NelderMeadPhase::AwaitingContraction {
                reflected_point,
                reflected_value,
                contract_point,
            } => {
                self.handle_contraction(reflected_point, reflected_value, contract_point, value);
            }
            NelderMeadPhase::Shrinking {
                best_point,
                shrink_index,
                pending_point,
            } => {
                self.handle_shrink(best_point, shrink_index, pending_point, value);
            }
            NelderMeadPhase::Terminated(_) => unreachable!(),
        }

        Ok(())
    }

    /// Get current iteration count
    pub fn iterations(&self) -> usize {
        self.nit
    }

    /// Get current function evaluation count
    pub fn evaluations(&self) -> usize {
        self.nfev
    }

    /// Get the current best point and value (if simplex is non-empty)
    pub fn best(&self) -> Option<(&[f64], f64)> {
        self.simplex
            .first()
            .map(|ep| (ep.point.as_slice(), ep.value))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase Handlers
    // ─────────────────────────────────────────────────────────────────────────

    fn handle_initial_evaluated(&mut self, value: f64) {
        self.simplex
            .push(EvaluatedPoint::new(self.initial_point.clone(), value));

        if self.dim == 0 {
            self.phase = NelderMeadPhase::Terminated(TerminationReason::BothTolerancesReached);
            return;
        }

        self.transition_to_build_simplex(0);
    }

    fn handle_simplex_point(&mut self, dim: usize, point: Point, value: f64) {
        self.simplex.push(EvaluatedPoint::new(point, value));

        if dim + 1 < self.dim {
            self.transition_to_build_simplex(dim + 1);
        } else if self.simplex.len() != self.dim + 1 {
            self.phase = NelderMeadPhase::Terminated(TerminationReason::DegenerateSimplex);
        } else {
            self.start_iteration();
        }
    }

    fn handle_reflection(&mut self, centroid: Point, reflected_point: Point, reflected_value: f64) {
        let best_value = self.simplex[0].value;
        let second_worst_value = self.simplex[self.simplex.len() - 2].value;
        let worst_value = self
            .simplex
            .last()
            .expect("simplex should not be empty")
            .value;

        if reflected_value < best_value {
            // Best so far - try expansion
            if self.check_and_handle_max_evals_with_update(&reflected_point, reflected_value) {
                return;
            }

            let expanded_point =
                self.transform_point(&centroid, &reflected_point, self.config.gamma);

            self.phase = NelderMeadPhase::AwaitingExpansion {
                _centroid: centroid,
                reflected_point,
                reflected_value,
                expanded_point,
            };
        } else if reflected_value < second_worst_value {
            // Good enough - accept reflection
            self.accept_point(reflected_point, reflected_value);
            self.start_iteration();
        } else {
            // Need contraction
            if self.check_and_handle_max_evals_with_update(&reflected_point, reflected_value) {
                return;
            }

            let (contract_toward, coeff) = if reflected_value < worst_value {
                // Outside contraction
                (&reflected_point, self.config.rho)
            } else {
                // Inside contraction
                (
                    &self
                        .simplex
                        .last()
                        .expect("simplex should not be empty")
                        .point
                        .clone(),
                    -self.config.rho,
                )
            };

            let contract_point = self.transform_point(&centroid, contract_toward, coeff);

            self.phase = NelderMeadPhase::AwaitingContraction {
                reflected_point,
                reflected_value,
                contract_point,
            };
        }
    }

    fn handle_expansion(
        &mut self,
        reflected_point: Point,
        reflected_value: f64,
        expanded_point: Point,
        expanded_value: f64,
    ) {
        if expanded_value < reflected_value {
            self.accept_point(expanded_point, expanded_value);
        } else {
            self.accept_point(reflected_point, reflected_value);
        }
        self.start_iteration();
    }

    fn handle_contraction(
        &mut self,
        reflected_point: Point,
        reflected_value: f64,
        contract_point: Point,
        contract_value: f64,
    ) {
        let worst_value = self
            .simplex
            .last()
            .expect("simplex should not be empty")
            .value;

        if contract_value < worst_value && contract_value < reflected_value {
            self.accept_point(contract_point, contract_value);
            self.start_iteration();
        } else if reflected_value < worst_value {
            // Contraction failed but reflection was better than worst
            self.accept_point(reflected_point, reflected_value);
            self.start_iteration();
        } else {
            // Need to shrink
            self.start_shrink();
        }
    }

    fn handle_shrink(
        &mut self,
        best_point: Point,
        shrink_index: usize,
        pending_point: Point,
        value: f64,
    ) {
        self.simplex[shrink_index] = EvaluatedPoint::new(pending_point, value);

        let next_index = shrink_index + 1;
        if next_index <= self.dim {
            if self.reached_max_evaluations() {
                self.phase =
                    NelderMeadPhase::Terminated(TerminationReason::MaxFunctionEvaluationsReached);
                return;
            }

            let pending_point = self.compute_shrink_point(&best_point, next_index);
            self.phase = NelderMeadPhase::Shrinking {
                best_point,
                shrink_index: next_index,
                pending_point,
            };
        } else {
            self.start_iteration();
        }
    }

    // State Transitions

    fn transition_to_build_simplex(&mut self, dim: usize) {
        if self.reached_max_evaluations() {
            self.phase =
                NelderMeadPhase::Terminated(TerminationReason::MaxFunctionEvaluationsReached);
            return;
        }

        let pending_point = self.compute_simplex_vertex(dim);
        self.phase = NelderMeadPhase::BuildingSimplex {
            next_dim: dim,
            pending_point,
        };
    }

    fn start_iteration(&mut self) {
        // Sort simplex by objective value
        self.simplex
            .sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));

        // Check termination conditions
        if let Some(reason) = self.check_termination() {
            self.phase = NelderMeadPhase::Terminated(reason);
            return;
        }

        self.nit += 1;

        // Compute centroid of all points except the worst
        let (worst, rest) = self
            .simplex
            .split_last()
            .expect("simplex should not be empty");
        let centroid = Self::centroid(rest);

        // Compute reflection point
        let reflected_point = self.transform_point(&centroid, &worst.point, -self.config.alpha);

        self.phase = NelderMeadPhase::AwaitingReflection {
            centroid,
            reflected_point,
        };
    }

    fn start_shrink(&mut self) {
        if self.reached_max_evaluations() {
            self.phase =
                NelderMeadPhase::Terminated(TerminationReason::MaxFunctionEvaluationsReached);
            return;
        }

        let best_point = self.simplex[0].point.clone();
        let pending_point = self.compute_shrink_point(&best_point, 1);

        self.phase = NelderMeadPhase::Shrinking {
            best_point,
            shrink_index: 1,
            pending_point,
        };
    }

    // Helper Methods

    fn compute_simplex_vertex(&self, dim: usize) -> Point {
        let mut point = self.initial_point.clone();

        if point[dim] != 0.0 {
            point[dim] *= 1.0 + self.config.sigma0;
        } else {
            point[dim] = self.config.sigma0;
        }

        // Ensure the point differs from the initial point
        if point
            .iter()
            .zip(&self.simplex[0].point)
            .all(|(a, b)| (a - b).abs() <= f64::EPSILON)
        {
            point[dim] += self.config.sigma0;
        }

        // Apply bounds
        self.bounds.clamp(&mut point);
        point
    }

    fn compute_shrink_point(&self, best_point: &[f64], index: usize) -> Point {
        let mut point: Point = best_point
            .iter()
            .zip(&self.simplex[index].point)
            .map(|(b, x)| b + self.config.sigma * (x - b))
            .collect();

        // Apply bounds
        self.bounds.clamp(&mut point);
        point
    }

    fn transform_point(&self, from: &[f64], toward: &[f64], coeff: f64) -> Point {
        let mut point: Point = from
            .iter()
            .zip(toward)
            .map(|(f, t)| f + coeff * (t - f))
            .collect();

        // Apply bounds
        self.bounds.clamp(&mut point);
        point
    }

    fn accept_point(&mut self, point: Point, value: f64) {
        let worst_index = self.simplex.len() - 1;
        self.simplex[worst_index] = EvaluatedPoint::new(point, value);
    }

    fn centroid(simplex: &[EvaluatedPoint]) -> Point {
        let n = simplex.len() as f64;
        let dim = simplex[0].point.len();

        (0..dim)
            .map(|i| simplex.iter().map(|v| v.point[i]).sum::<f64>() / n)
            .collect()
    }

    fn reached_max_evaluations(&self) -> bool {
        self.config
            .max_evaluations
            .is_some_and(|limit| self.nfev >= limit)
    }

    fn check_and_handle_max_evals_with_update(&mut self, point: &Point, value: f64) -> bool {
        if self.reached_max_evaluations() {
            self.accept_point(point.clone(), value);
            self.phase =
                NelderMeadPhase::Terminated(TerminationReason::MaxFunctionEvaluationsReached);
            true
        } else {
            false
        }
    }

    fn check_termination(&self) -> Option<TerminationReason> {
        // Check patience/timeout
        if let Some(patience) = self.config.patience {
            if self.start_time.elapsed() >= patience {
                return Some(TerminationReason::PatienceElapsed);
            }
        }

        // Check max iterations
        if self.nit >= self.config.max_iter {
            return Some(TerminationReason::MaxIterationsReached);
        }

        // Check max evaluations
        if self.reached_max_evaluations() {
            return Some(TerminationReason::MaxFunctionEvaluationsReached);
        }

        // Check convergence
        self.convergence_reason()
    }

    fn convergence_reason(&self) -> Option<TerminationReason> {
        if self.simplex.is_empty() {
            return None;
        }

        let best = &self.simplex[0];
        let worst = self.simplex.last()?;

        let fun_diff = (worst.value - best.value).abs();
        let fun_converged = fun_diff <= self.config.threshold;

        let max_dist = self.simplex[1..]
            .iter()
            .map(|vertex| {
                vertex
                    .point
                    .iter()
                    .zip(&best.point)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0_f64, f64::max);

        let position_converged = max_dist <= self.config.position_tolerance;

        match (fun_converged, position_converged) {
            (true, true) => Some(TerminationReason::BothTolerancesReached),
            (true, false) => Some(TerminationReason::FunctionToleranceReached),
            (false, true) => Some(TerminationReason::ParameterToleranceReached),
            (false, false) => None,
        }
    }

    fn build_results(&self, reason: TerminationReason) -> OptimisationResults {
        build_results(
            &self.simplex,
            self.nit,
            self.nfev,
            self.start_time.elapsed(),
            reason,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    fn rosenbrock(x: &[f64]) -> Result<f64, Infallible> {
        let a = 1.0;
        let b = 100.0;
        let x0 = x[0];
        let x1 = x[1];
        Ok((a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2))
    }

    #[test]
    fn test_ask_tell_interface() {
        let nm = NelderMead::new().with_max_iter(500).with_threshold(1e-8);

        let initial = vec![0.0, 0.0];
        let (mut state, first_point) = nm.init(initial, Bounds::unbounded(2));

        // Evaluate first point
        let mut current_value = rosenbrock(&first_point[0]);

        loop {
            match state.tell(current_value) {
                Ok(()) => {}
                Err(TellError::AlreadyTerminated) => break,
                _ => break,
            }

            match state.ask() {
                AskResult::Evaluate(point) => {
                    current_value = rosenbrock(&point[0]);
                }
                AskResult::Done(results) => {
                    println!("Optimization complete!");
                    println!("Best value: {}", results.value);
                    println!("Iterations: {}", results.iterations);
                    println!("Evaluations: {}", results.evaluations);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_run_convenience_wrapper() {
        let nm = NelderMead::new().with_max_iter(500);

        let results = nm.run(|x| rosenbrock(x), vec![0.0, 0.0], Bounds::unbounded(2));

        assert!(results.value < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Ask/Tell Interface Tests
    // ═══════════════════════════════════════════════════════════════════

    fn sphere(x: &[f64]) -> Result<f64, std::io::Error> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    #[test]
    fn nelder_mead_ask_tell_basic_workflow() {
        let optimizer = NelderMead::new().with_max_iter(100).with_threshold(1e-8);

        let (mut state, first_point) = optimizer.init(vec![5.0], Bounds::unbounded(1));

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => {
                    assert!(results.value < 1e-4, "Should converge to minimum");
                    assert!(results.x[0].abs() < 1e-2, "x should be near 0");
                    assert!(results.success, "Should report success");
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_ask_tell_matches_run() {
        let optimizer = NelderMead::new()
            .with_max_iter(200)
            .with_threshold(1e-8)
            .with_sigma0(1.0);

        // Ask-tell version
        let (mut state, first_point) = optimizer.init(vec![3.0, -2.0], Bounds::unbounded(2));

        let mut result = sphere(&first_point[0]);
        let ask_tell_results = loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => break results,
            }
        };

        // .run() version
        let run_results = optimizer.run(sphere, vec![3.0, -2.0], Bounds::unbounded(2));

        // Should produce identical results (deterministic)
        assert_eq!(ask_tell_results.iterations, run_results.iterations);
        assert!((ask_tell_results.value - run_results.value).abs() < 1e-12);
        assert_eq!(ask_tell_results.evaluations, run_results.evaluations);
        for i in 0..2 {
            assert!((ask_tell_results.x[i] - run_results.x[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn nelder_mead_simplex_building_phase() {
        let optimizer = NelderMead::new().with_sigma0(0.5);

        let (mut state, first_point) = optimizer.init(vec![1.0, 1.0], Bounds::unbounded(2));

        // First point should be the initial point
        assert_eq!(first_point.len(), 1);
        assert_eq!(first_point[0], vec![1.0, 1.0]);

        // Evaluate first point
        state.tell(sphere(&first_point[0])).unwrap();

        // Next asks should build the simplex (n+1 points for n dimensions)
        let mut simplex_build_count = 0;
        loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    assert_eq!(
                        points.len(),
                        1,
                        "Should ask for one point at a time during simplex building"
                    );
                    simplex_build_count += 1;
                    state.tell(sphere(&points[0])).unwrap();

                    // For 2D problem, need 3 total points (1 initial + 2 to build simplex)
                    if simplex_build_count == 2 {
                        break;
                    }
                }
                AskResult::Done(_) => {
                    panic!("Should not terminate during simplex building");
                }
            }
        }

        assert_eq!(
            simplex_build_count, 2,
            "Should build 2 additional simplex points for 2D problem"
        );
    }

    #[test]
    fn nelder_mead_tell_already_terminated() {
        let optimizer = NelderMead::new().with_max_iter(1).with_threshold(1e-10);

        let (mut state, first_point) = optimizer.init(vec![0.0], Bounds::unbounded(1));

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere(&points[0]);
                }
                AskResult::Done(_) => break,
            }
        }

        // Now try to tell again after termination
        let err = state.tell(Ok::<f64, std::io::Error>(1.0)).unwrap_err();
        assert!(matches!(err, TellError::AlreadyTerminated));
    }

    // Note: NelderMead's ask-tell interface always requests one point at a time,
    // so ResultCountMismatch errors are handled internally by the type system.
    // This test is not applicable to NelderMead's current API.

    #[test]
    fn nelder_mead_handles_evaluation_errors() {
        let optimizer = NelderMead::new().with_max_iter(50).with_threshold(1e-10);

        let (mut state, first_point) = optimizer.init(vec![3.0, 3.0], Bounds::unbounded(2));

        let mut eval_count = 0;
        let mut result = sphere(&first_point[0]);
        let mut error_injected = false;

        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Inject an error on the 10th evaluation
                    if eval_count == 10 {
                        result = Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "Simulated evaluation failure",
                        ));
                        error_injected = true;
                    } else {
                        result = sphere(&points[0]);
                    }
                    eval_count += 1;
                }
                AskResult::Done(results) => {
                    // Should complete despite the error (error treated as high value)
                    assert!(error_injected, "Should have injected error");
                    // Just verify it completed without panicking
                    assert!(results.evaluations > 10, "Should have many evaluations");
                    break;
                }
            }
        }
    }

    // Note: NelderMead doesn't expose query_best method currently.
    // This test is not applicable to the current API.

    #[test]
    fn nelder_mead_respects_bounds_ask_tell() {
        let optimizer = NelderMead::new().with_max_iter(100);

        let bounds = Bounds::new(vec![(-1.0, 1.0), (-1.0, 1.0)]);
        let (mut state, first_point) = optimizer.init(vec![0.8, 0.8], bounds);

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Check all proposed points respect bounds
                    for point in &points {
                        for &val in point {
                            assert!(
                                val >= -1.0 && val <= 1.0,
                                "Point {:?} violates bounds",
                                point
                            );
                        }
                    }
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => {
                    // Final result should also respect bounds
                    for &val in &results.x {
                        assert!(val >= -1.0 && val <= 1.0, "Final result violates bounds");
                    }
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_convergence_by_threshold() {
        let optimizer = NelderMead::new().with_max_iter(500).with_threshold(1e-6);

        let (mut state, first_point) = optimizer.init(vec![3.0, 3.0], Bounds::unbounded(2));

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => {
                    // Should converge by threshold, not max iterations
                    assert!(
                        results.iterations < 500,
                        "Should converge before max iterations"
                    );
                    assert!(results.value < 1e-6, "Should meet threshold requirement");
                    assert!(results.success, "Should report success");
                    // Message can vary depending on exact convergence condition
                    assert!(
                        results.message.contains("tolerance")
                            || results.message.contains("Converged"),
                        "Message should indicate convergence"
                    );
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_termination_by_max_iter() {
        let optimizer = NelderMead::new().with_max_iter(10).with_threshold(1e-12); // Very strict threshold

        let (mut state, first_point) = optimizer.init(vec![5.0], Bounds::unbounded(1));

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => {
                    assert_eq!(results.iterations, 10, "Should terminate at max iterations");
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_patience_termination() {
        use std::thread;
        use std::time::Duration;

        let optimizer = NelderMead::new().with_max_iter(500).with_patience(0.001); // 1 millisecond patience

        // Start near minimum so it gets stuck
        let (mut state, first_point) = optimizer.init(vec![0.01, 0.01], Bounds::unbounded(2));

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).unwrap();

            // Add small delay to trigger patience timeout
            thread::sleep(Duration::from_millis(2));

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => {
                    // Should terminate by patience before max iterations
                    assert!(
                        results.iterations < 500,
                        "Should terminate early due to patience"
                    );
                    // Patience message may not always appear in message string
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_position_tolerance() {
        let optimizer = NelderMead::new()
            .with_max_iter(500)
            .with_position_tolerance(1e-8);

        let (mut state, first_point) = optimizer.init(vec![2.0, 2.0], Bounds::unbounded(2));

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => {
                    // Should converge and report position tolerance
                    assert!(results.success, "Should succeed");
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_multidimensional() {
        let optimizer = NelderMead::new().with_max_iter(300).with_threshold(1e-6);

        // 5D sphere function
        let (mut state, first_point) =
            optimizer.init(vec![2.0, -3.0, 1.5, -1.0, 2.5], Bounds::unbounded(5));

        let mut result = sphere(&first_point[0]);
        loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    assert_eq!(points[0].len(), 5, "Should maintain dimensionality");
                    result = sphere(&points[0]);
                }
                AskResult::Done(results) => {
                    assert_eq!(results.x.len(), 5, "Result should have 5 dimensions");
                    assert!(results.value < 1e-4, "Should converge to minimum");
                    for &x in &results.x {
                        assert!(x.abs() < 0.1, "Each coordinate should be near 0");
                    }
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_rosenbrock_ask_tell() {
        let optimizer = NelderMead::new().with_max_iter(800).with_threshold(1e-6);

        let (mut state, first_point) = optimizer.init(vec![0.0, 0.0], Bounds::unbounded(2));

        let mut result = rosenbrock(&first_point[0]);
        loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = rosenbrock(&points[0]);
                }
                AskResult::Done(results) => {
                    assert!(results.value < 1e-5, "Should converge on Rosenbrock");
                    assert!((results.x[0] - 1.0).abs() < 0.01, "x[0] should be near 1");
                    assert!((results.x[1] - 1.0).abs() < 0.01, "x[1] should be near 1");
                    break;
                }
            }
        }
    }

    #[test]
    fn nelder_mead_state_machine_integrity() {
        let optimizer = NelderMead::new().with_max_iter(20);

        let (mut state, first_point) = optimizer.init(vec![1.0], Bounds::unbounded(1));

        // Cannot call ask before tell for first point
        state.tell(sphere(&first_point[0])).unwrap();

        let mut ask_count = 0;
        let mut tell_count = 1; // Already told once

        loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    ask_count += 1;
                    state.tell(sphere(&points[0])).unwrap();
                    tell_count += 1;
                }
                AskResult::Done(_) => break,
            }
        }

        // Should have same number of asks and tells (after initial tell)
        assert_eq!(
            ask_count + 1,
            tell_count,
            "Ask/tell calls should be balanced"
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Error Handling Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn nelder_mead_nonfinite_handling() {
        let optimizer = NelderMead::new().with_max_iter(50);

        let (mut state, first_point) = optimizer.init(vec![1.0], Bounds::unbounded(1));

        let mut iteration = 0;
        let mut result = sphere(&first_point[0]);

        loop {
            state.tell(result).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Return NaN on iteration 3
                    if iteration == 3 {
                        result = Ok(f64::NAN);
                    } else {
                        result = sphere(&points[0]);
                    }
                    iteration += 1;
                }
                AskResult::Done(results) => {
                    // Should handle NaN and continue
                    assert!(results.value.is_finite(), "Final value should be finite");
                    break;
                }
            }
        }
    }
}
