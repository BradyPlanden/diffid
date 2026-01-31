use crate::common::{AskResult, Bounds, Point};
use crate::errors::{EvaluationError, TellError};
use crate::optimisers::{
    build_results, EvaluatedPoint, GradientEvaluation, OptimisationResults, TerminationReason,
};
use std::error::Error as StdError;
use std::time::{Duration, Instant};

/// Configuration for the Adam optimiser
#[must_use]
#[derive(Clone, Debug)]
pub struct Adam {
    max_iter: usize,
    step_size: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    threshold: f64,
    gradient_threshold: Option<f64>,
    patience: Option<Duration>,
}

impl Adam {
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            step_size: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            threshold: 1e-6,
            gradient_threshold: None, // Uses threshold if None
            patience: None,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_step_size(mut self, step_size: f64) -> Self {
        if step_size.is_finite() && step_size > 0.0 {
            self.step_size = step_size;
        }
        self
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        if (1e-10..1.0).contains(&beta1) && (1e-10..1.0).contains(&beta2) {
            self.beta1 = beta1;
            self.beta2 = beta2;
        }
        self
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        if eps.is_finite() && eps > 0.0 {
            self.eps = eps;
        }
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.max(0.0);
        self
    }

    pub fn with_gradient_threshold(mut self, threshold: f64) -> Self {
        self.gradient_threshold = Some(threshold.max(0.0));
        self
    }

    pub fn with_patience(mut self, patience: f64) -> Self {
        self.patience = Some(Duration::from_secs_f64(patience));
        self
    }

    /// Get the effective gradient threshold
    fn gradient_threshold(&self) -> f64 {
        self.gradient_threshold.unwrap_or(self.threshold)
    }

    /// Initialize the optimisation state
    ///
    /// Returns the state and the first point to evaluate
    pub fn init(&self, initial: Point, bounds: Bounds) -> (AdamState, Point) {
        let dim = initial.len();
        let mut initial_point = initial;
        bounds.clamp(&mut initial_point);

        let state = AdamState::new(self.clone(), initial_point.clone(), bounds, dim);
        (state, initial_point)
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new()
    }
}

/// The current phase of the Adam algorithm
#[derive(Clone, Debug)]
pub enum AdamPhase {
    /// Waiting for point evaluation with gradient
    AwaitingEvaluation { pending_point: Point },

    /// Algorithm has terminated
    Terminated(TerminationReason),
}

/// State of the Adam momentum estimates
#[derive(Clone, Debug)]
struct MomentumState {
    /// First moment estimate
    m: Vec<f64>,
    /// Second moment estimate
    v: Vec<f64>,
    /// beta1^t for bias correction
    beta1_pow: f64,
    /// beta2^t for bias correction
    beta2_pow: f64,
}

impl MomentumState {
    fn new(dim: usize) -> Self {
        Self {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            beta1_pow: 1.0,
            beta2_pow: 1.0,
        }
    }

    /// Update momentum estimates and compute parameter update
    fn compute_update(
        &mut self,
        gradient: &[f64],
        step_size: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) -> Vec<f64> {
        // Update power terms for bias correction
        self.beta1_pow *= beta1;
        self.beta2_pow *= beta2;

        let bias_correction1 = (1.0 - self.beta1_pow).max(1e-12);
        let bias_correction2 = (1.0 - self.beta2_pow).max(1e-12);

        let mut update = Vec::with_capacity(gradient.len());

        for (i, g) in gradient.iter().enumerate() {
            // Update biased first moment estimate
            self.m[i] = beta1.mul_add(self.m[i], (1.0 - beta1) * g);
            // Update biased second moment estimate
            self.v[i] = beta2.mul_add(self.v[i], (1.0 - beta2) * g * g);

            // Compute bias-corrected estimates
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Compute update
            let denom = v_hat.sqrt() + eps;
            update.push(step_size * m_hat / denom);
        }

        update
    }
}

/// Runtime state of the Adam optimiser
pub struct AdamState {
    config: Adam,
    bounds: Bounds,
    dim: usize,

    // Current position
    x: Point,

    // Adam momentum state
    momentum: MomentumState,

    // Current phase
    phase: AdamPhase,

    // Tracking
    iterations: usize,
    evaluations: usize,
    start_time: Instant,
    history: Vec<EvaluatedPoint>,
    prev_cost: Option<f64>,
}

impl AdamState {
    fn new(config: Adam, initial_point: Point, bounds: Bounds, dim: usize) -> Self {
        Self {
            config,
            bounds,
            dim,
            x: initial_point.clone(),
            momentum: MomentumState::new(dim),
            phase: AdamPhase::AwaitingEvaluation {
                pending_point: initial_point,
            },
            iterations: 0,
            evaluations: 0,
            start_time: Instant::now(),
            history: Vec::new(),
            prev_cost: None,
        }
    }

    /// Get the next point to evaluate, or the final result if optimisation is complete
    pub fn ask(&self) -> AskResult<OptimisationResults> {
        match &self.phase {
            AdamPhase::Terminated(reason) => AskResult::Done(self.build_results(reason.clone())),
            AdamPhase::AwaitingEvaluation { pending_point } => {
                AskResult::Evaluate(vec![pending_point.clone()])
            }
        }
    }

    /// Report the evaluation result (value and gradient) for the last point from `ask()`
    ///
    /// Pass `Err` if the objective function failed to evaluate
    pub fn tell<T, E>(&mut self, result: T) -> Result<(), TellError>
    where
        T: TryInto<GradientEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        if matches!(self.phase, AdamPhase::Terminated(_)) {
            return Err(TellError::AlreadyTerminated);
        }

        // Convert to evaluation result
        let eval = match result.try_into() {
            Ok(e) => e,
            Err(e) => {
                let err: EvaluationError = e.into();
                self.history
                    .push(EvaluatedPoint::new(self.x.clone(), f64::NAN));
                self.phase = AdamPhase::Terminated(TerminationReason::FunctionEvaluationFailed(
                    format!("{err}"),
                ));
                return Ok(());
            }
        };

        // Validate gradient dimension
        if eval.gradient.len() != self.dim {
            return Err(TellError::GradientDimensionMismatch {
                expected: self.dim,
                got: eval.gradient.len(),
            });
        }

        self.evaluations += 1;

        // Take ownership of current phase
        let phase = std::mem::replace(
            &mut self.phase,
            AdamPhase::Terminated(TerminationReason::MaxIterationsReached),
        );

        match phase {
            AdamPhase::AwaitingEvaluation { pending_point } => {
                self.handle_evaluation(pending_point, eval);
            }
            AdamPhase::Terminated(_) => unreachable!(),
        }

        Ok(())
    }

    /// Get current iteration count
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Get current function evaluation count
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Get the current best point and value
    pub fn best(&self) -> Option<(&[f64], f64)> {
        self.history
            .iter()
            .filter(|p| p.value.is_finite())
            .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap())
            .map(|ep| (ep.point.as_slice(), ep.value))
    }

    /// Get the current position
    pub fn current_position(&self) -> &[f64] {
        &self.x
    }

    /// Get the current momentum state (m, v)
    pub fn momentum(&self) -> (&[f64], &[f64]) {
        (&self.momentum.m, &self.momentum.v)
    }

    // Phase Handler
    fn handle_evaluation(&mut self, _pending_point: Point, eval: GradientEvaluation) {
        let GradientEvaluation { value, gradient } = eval;

        // Validate gradient values
        if !gradient.iter().all(|g| g.is_finite()) {
            self.history
                .push(EvaluatedPoint::new(self.x.clone(), value));
            self.phase = AdamPhase::Terminated(TerminationReason::FunctionEvaluationFailed(
                "Gradient contained non-finite values".to_string(),
            ));
            return;
        }

        // Record point in history
        self.history
            .push(EvaluatedPoint::new(self.x.clone(), value));

        // Check gradient convergence
        let grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm <= self.config.gradient_threshold() {
            self.phase = AdamPhase::Terminated(TerminationReason::GradientToleranceReached);
            return;
        }

        // Check cost convergence
        if let Some(prev_cost) = self.prev_cost {
            if (prev_cost - value).abs() < self.config.threshold {
                self.phase = AdamPhase::Terminated(TerminationReason::FunctionToleranceReached);
                return;
            }
        }
        self.prev_cost = Some(value);

        // Check max iterations (before computing next step)
        if self.iterations >= self.config.max_iter {
            self.phase = AdamPhase::Terminated(TerminationReason::MaxIterationsReached);
            return;
        }

        // Check patience/timeout
        if let Some(patience) = self.config.patience {
            if self.start_time.elapsed() >= patience {
                self.phase = AdamPhase::Terminated(TerminationReason::PatienceElapsed);
                return;
            }
        }

        // Compute parameter update
        let update = self.momentum.compute_update(
            &gradient,
            self.config.step_size,
            self.config.beta1,
            self.config.beta2,
            self.config.eps,
        );

        // Apply update
        for (xi, delta) in self.x.iter_mut().zip(update.iter()) {
            *xi -= delta;
        }

        // Apply bounds
        self.bounds.clamp(&mut self.x);

        self.iterations += 1;

        // Prepare for next evaluation
        self.phase = AdamPhase::AwaitingEvaluation {
            pending_point: self.x.clone(),
        };
    }

    // fn apply_bounds_to(&self, point: &mut Point) {
    //     point.apply_bounds(self.bounds.as_ref());
    // }

    fn build_results(&self, reason: TerminationReason) -> OptimisationResults {
        let points = if self.history.is_empty() {
            // If we never evaluated, create a dummy point
            vec![EvaluatedPoint::new(self.x.clone(), f64::NAN)]
        } else {
            self.history.clone()
        };

        build_results(
            &points,
            self.iterations,
            self.evaluations,
            self.start_time.elapsed(),
            reason,
            None,
        )
    }
}

// Convenience wrapper
impl Adam {
    /// Run optimisation using a closure for evaluation
    ///
    /// The closure should return `(value, gradient)` for a given point
    pub fn run<F, R, E>(
        &self,
        mut objective: F,
        initial: Point,
        bounds: Bounds,
    ) -> OptimisationResults
    where
        F: FnMut(&[f64]) -> R,
        R: TryInto<GradientEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        let (mut state, first_point) = self.init(initial, bounds);
        let mut result = objective(&first_point);

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

        match state.ask() {
            AskResult::Done(results) => results,
            _ => panic!("Unexpected state after tell error"),
        }
    }

    /// Run optimisation with numerical gradient approximation
    ///
    /// Uses central differences to approximate the gradient
    pub fn run_with_numerical_gradient<F, E>(
        &self,
        mut objective: F,
        initial: Point,
        bounds: Bounds,
        epsilon: f64,
    ) -> OptimisationResults
    where
        F: FnMut(&[f64]) -> Result<f64, E>,
        E: StdError + Send + Sync + 'static,
    {
        self.run(
            |x| -> Result<(f64, Vec<f64>), E> {
                let value = objective(x)?;
                let mut gradient = vec![0.0; x.len()];
                let mut x_plus = x.to_vec();
                let mut x_minus = x.to_vec();

                for i in 0..x.len() {
                    x_plus[i] = x[i] + epsilon;
                    x_minus[i] = x[i] - epsilon;

                    let f_plus = objective(&x_plus)?;
                    let f_minus = objective(&x_minus)?;

                    gradient[i] = (f_plus - f_minus) / (2.0 * epsilon);

                    x_plus[i] = x[i];
                    x_minus[i] = x[i];
                }

                Ok((value, gradient))
            },
            initial,
            bounds,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock function with analytical gradient
    fn rosenbrock_infallible(x: &[f64]) -> (f64, Vec<f64>) {
        let a = 1.0;
        let b = 100.0;
        let x0 = x[0];
        let x1 = x[1];

        let value = (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2);

        let grad = vec![
            -2.0 * (a - x0) - 4.0 * b * x0 * (x1 - x0.powi(2)),
            2.0 * b * (x1 - x0.powi(2)),
        ];

        (value, grad)
    }

    /// Infallible sphere function
    fn sphere_infallible(x: &[f64]) -> (f64, Vec<f64>) {
        let value: f64 = x.iter().map(|xi| xi * xi).sum();
        let grad: Vec<f64> = x.iter().map(|xi| 2.0 * xi).collect();
        (value, grad)
    }

    /// fallible sphere function
    fn sphere_fallible(x: &[f64]) -> Result<(f64, Vec<f64>), std::io::Error> {
        let value: f64 = x.iter().map(|xi| xi * xi).sum();
        let grad: Vec<f64> = x.iter().map(|xi| 2.0 * xi).collect();
        Ok((value, grad))
    }

    #[test]
    fn test_ask_tell_fallible() {
        let adam = Adam::new()
            .with_max_iter(1000)
            .with_step_size(0.1)
            .with_threshold(1e-6);

        let initial = vec![0.0, 0.0];
        let (mut state, first_point) = adam.init(initial, Bounds::unbounded(2));
        let mut val_grad = sphere_fallible(&first_point);

        loop {
            if state.tell(val_grad).is_err() {
                break;
            }

            match state.ask() {
                AskResult::Evaluate(point) => {
                    val_grad = sphere_fallible(&point[0]);
                }
                AskResult::Done(results) => {
                    assert!(results.value < 1e-4);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_ask_tell_infallible() {
        let adam = Adam::new()
            .with_max_iter(1000)
            .with_step_size(0.1)
            .with_threshold(1e-6);

        let initial = vec![0.0, 0.0];
        let (mut state, first_point) = adam.init(initial, Bounds::unbounded(2));
        let mut val_grad = sphere_infallible(&first_point);

        loop {
            if state.tell(val_grad).is_err() {
                break;
            }

            match state.ask() {
                AskResult::Evaluate(point) => {
                    val_grad = sphere_infallible(&point[0]);
                }
                AskResult::Done(results) => {
                    assert!(results.value < 1e-4);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_run_convenience_wrapper() {
        let adam = Adam::new().with_max_iter(1000).with_step_size(0.1);

        let results = adam.run(sphere_infallible, vec![5.0, 5.0], Bounds::unbounded(2));

        println!("Final value: {}", results.value);
        println!("Iterations: {}", results.iterations);
        assert!(results.value < 1e-3);
    }

    #[test]
    fn test_rosenbrock() {
        let adam = Adam::new()
            .with_max_iter(10000)
            .with_step_size(0.001)
            .with_threshold(1e-8);

        let results = adam.run(rosenbrock_infallible, vec![0.0, 0.0], Bounds::unbounded(2));

        println!("Rosenbrock result: {}", results.value);
        println!("Iterations: {}", results.iterations);
        // Rosenbrock is harder - just check we made progress
        assert!(results.value < 1.0);
    }

    #[test]
    fn test_gradient_dimension_mismatch() {
        let adam = Adam::new();
        let (mut state, _) = adam.init(vec![1.0, 2.0], Bounds::unbounded(2));

        // Wrong gradient dimension
        let bad_eval = GradientEvaluation::new(1.0, vec![0.1]); // Should be 2 elements
        let result = state.tell(bad_eval);

        assert!(matches!(
            result,
            Err(TellError::GradientDimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn adam_ask_tell_basic_workflow() {
        let optimiser = Adam::new()
            .with_max_iter(500)
            .with_step_size(0.1)
            .with_threshold(1e-8);

        let (mut state, first_point) = optimiser.init(vec![3.0, -2.0], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    assert_eq!(points.len(), 1);
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(results) => {
                    assert!(results.value < 1e-4);
                    assert!(results.x[0].abs() < 1e-2);
                    assert!(results.x[1].abs() < 1e-2);
                    break;
                }
            }
        }
    }

    #[test]
    fn adam_ask_tell_matches_run() {
        let optimiser = Adam::new()
            .with_max_iter(100)
            .with_step_size(0.1)
            .with_threshold(1e-6);

        // Ask-tell interface
        let (mut state, first_point) = optimiser.clone().init(vec![2.0, 2.0], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        loop {
            state.tell(result).expect("tell should succeed");
            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(results) => {
                    let ask_tell_value = results.value;

                    // Run interface
                    let run_results =
                        optimiser.run(sphere_infallible, vec![2.0, 2.0], Bounds::unbounded(2));

                    // Should get very similar results
                    assert!((ask_tell_value - run_results.value).abs() < 1e-10);
                    break;
                }
            }
        }
    }

    #[test]
    fn adam_momentum_accumulation() {
        let optimiser = Adam::new().with_max_iter(100).with_step_size(0.01);

        let (mut state, first_point) = optimiser.init(vec![1.0, 1.0], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        // Initial momentum should be zero
        let (m0, v0) = state.momentum();
        assert!(m0.iter().all(|&x| x == 0.0));
        assert!(v0.iter().all(|&x| x == 0.0));

        // After first update, momentum should be non-zero
        state.tell(result).expect("tell should succeed");
        match state.ask() {
            AskResult::Evaluate(points) => {
                let (m1, v1) = state.momentum();
                assert!(
                    m1.iter().any(|&x| x != 0.0),
                    "First moment should be updated"
                );
                assert!(
                    v1.iter().any(|&x| x != 0.0),
                    "Second moment should be updated"
                );

                result = sphere_infallible(&points[0]);
            }
            AskResult::Done(_) => return,
        }

        // After second update, momentum should have accumulated
        state.tell(result).expect("tell should succeed");
        let (m2, v2) = state.momentum();
        assert!(
            m2.iter()
                .zip(v2.iter())
                .all(|(&m, &v)| m != 0.0 && v != 0.0),
            "Momentum should continue accumulating"
        );
    }

    #[test]
    fn adam_bias_correction() {
        let optimiser = Adam::new()
            .with_max_iter(10)
            .with_step_size(0.1)
            .with_betas(0.9, 0.999);

        let (mut state, first_point) = optimiser.init(vec![1.0], Bounds::unbounded(1));
        let mut result = sphere_infallible(&first_point);

        let mut positions = vec![first_point[0]];

        // Collect positions during optimisation
        for _ in 0..5 {
            state.tell(result).expect("tell should succeed");
            match state.ask() {
                AskResult::Evaluate(points) => {
                    positions.push(points[0][0]);
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(_) => break,
            }
        }

        // Bias correction should prevent initial updates from being too small
        // Check that we're making reasonable progress in early iterations
        assert!(
            (positions[0] - positions[1]).abs() > 1e-4,
            "First step should be significant due to bias correction"
        );
    }

    #[test]
    fn adam_query_momentum_state() {
        let optimiser = Adam::new().with_max_iter(50).with_step_size(0.1);

        let (mut state, first_point) = optimiser.init(vec![2.0, -3.0], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        // Run a few iterations
        for _ in 0..10 {
            state.tell(result).expect("tell should succeed");
            match state.ask() {
                AskResult::Evaluate(points) => {
                    let (m, v) = state.momentum();

                    // Verify momentum state properties
                    assert_eq!(m.len(), 2, "First moment should match dimension");
                    assert_eq!(v.len(), 2, "Second moment should match dimension");
                    assert!(
                        m.iter().all(|&x| x.is_finite()),
                        "First moment should be finite"
                    );
                    assert!(
                        v.iter().all(|&x| x.is_finite() && x >= 0.0),
                        "Second moment should be non-negative"
                    );

                    // Verify current position is accessible
                    let pos = state.current_position();
                    assert_eq!(pos.len(), 2);
                    assert!(pos.iter().all(|&x| x.is_finite()));

                    // Verify counters
                    assert!(state.iterations() <= 50);
                    assert!(state.evaluations() <= 51);

                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(_) => break,
            }
        }
    }

    #[test]
    fn adam_numerical_gradient_accuracy() {
        let optimiser = Adam::new().with_max_iter(200).with_step_size(0.1);

        // Use numerical gradient approximation
        let results_numerical = optimiser.run_with_numerical_gradient(
            |x| -> Result<f64, std::io::Error> { Ok(x.iter().map(|xi| xi * xi).sum()) },
            vec![2.0, 2.0],
            Bounds::unbounded(2),
            1e-5,
        );

        // Use analytical gradient
        let results_analytical =
            optimiser.run(sphere_infallible, vec![2.0, 2.0], Bounds::unbounded(2));

        // Both should converge to similar values
        assert!(
            results_numerical.value < 1e-2,
            "Numerical gradient should converge"
        );
        assert!(
            results_analytical.value < 1e-3,
            "Analytical gradient should converge"
        );

        // Analytical should be more accurate with same iterations
        assert!(
            results_analytical.value <= results_numerical.value * 2.0,
            "Analytical gradient should be at least as good as numerical"
        );
    }

    #[test]
    fn adam_tell_already_terminated() {
        let optimiser = Adam::new().with_max_iter(1).with_step_size(0.1);

        let (mut state, first_point) = optimiser.init(vec![1.0], Bounds::unbounded(1));
        let mut result = sphere_infallible(&first_point);

        // First tell should succeed
        state.tell(result).expect("First tell should succeed");

        // Get next point to evaluate
        match state.ask() {
            AskResult::Evaluate(points) => {
                result = sphere_infallible(&points[0]);
                // Second tell should succeed and terminate
                state.tell(result).expect("Second tell should succeed");
            }
            AskResult::Done(_) => panic!("Should not terminate yet"),
        }

        // Should now be terminated
        match state.ask() {
            AskResult::Done(_) => {
                // Now try to tell again - should fail
                let third_result = sphere_infallible(&[0.5]);
                let tell_result = state.tell(third_result);

                assert!(
                    matches!(tell_result, Err(TellError::AlreadyTerminated)),
                    "Tell after termination should return AlreadyTerminated error"
                );
            }
            _ => panic!("Should have terminated after max_iter=1"),
        }
    }

    #[test]
    fn adam_nonfinite_gradient_handling() {
        let optimiser = Adam::new().with_max_iter(100).with_step_size(0.1);

        let (mut state, first_point) = optimiser.init(vec![1.0, 1.0], Bounds::unbounded(2));

        // First evaluation is normal
        let result1 = sphere_infallible(&first_point);
        state.tell(result1).expect("First tell should succeed");

        match state.ask() {
            AskResult::Evaluate(_points) => {
                // Second evaluation has non-finite gradient
                let bad_result = GradientEvaluation::new(1.0, vec![f64::NAN, 2.0]);
                state
                    .tell(bad_result)
                    .expect("tell should succeed even with bad gradient");

                // Should terminate due to non-finite gradient
                match state.ask() {
                    AskResult::Done(results) => {
                        assert!(matches!(
                            results.termination,
                            TerminationReason::FunctionEvaluationFailed(_)
                        ));
                    }
                    _ => panic!("Should have terminated due to non-finite gradient"),
                }
            }
            AskResult::Done(_) => panic!("Should not terminate after first evaluation"),
        }
    }

    #[test]
    fn adam_max_iterations_termination() {
        let optimiser = Adam::new()
            .with_max_iter(5)
            .with_step_size(0.001) // Very small step to prevent early convergence
            .with_threshold(1e-20); // Very tight threshold to prevent early convergence

        let (mut state, first_point) = optimiser.init(vec![10.0, 10.0], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        let mut iterations = 0;
        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    iterations += 1;
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(results) => {
                    assert_eq!(iterations, 5, "Should run exactly max_iter iterations");
                    assert!(matches!(
                        results.termination,
                        TerminationReason::MaxIterationsReached
                    ));
                    assert_eq!(results.iterations, 5);
                    break;
                }
            }
        }
    }

    #[test]
    fn adam_patience_timeout() {
        use std::time::Duration;

        let optimiser = Adam::new()
            .with_max_iter(10000)
            .with_step_size(0.1)
            .with_patience(0.001); // 1ms timeout

        let (mut state, first_point) = optimiser.init(vec![5.0, 5.0], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        // Add a small delay in the loop to ensure timeout is reached
        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    std::thread::sleep(Duration::from_micros(100)); // Small delay
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(results) => {
                    assert!(matches!(
                        results.termination,
                        TerminationReason::PatienceElapsed
                    ));
                    break;
                }
            }
        }
    }

    #[test]
    fn adam_function_tolerance_triggers() {
        let optimiser = Adam::new()
            .with_max_iter(1000)
            .with_step_size(0.1)
            .with_threshold(1e-4); // Loose threshold for function value change

        let (mut state, first_point) = optimiser.init(vec![0.001, 0.001], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(results) => {
                    // Should converge due to function tolerance
                    assert!(
                        matches!(
                            results.termination,
                            TerminationReason::FunctionToleranceReached
                        ) || matches!(
                            results.termination,
                            TerminationReason::GradientToleranceReached
                        )
                    );
                    break;
                }
            }
        }
    }

    #[test]
    fn adam_gradient_tolerance_triggers() {
        let optimiser = Adam::new()
            .with_max_iter(1000)
            .with_step_size(0.1)
            .with_gradient_threshold(0.5); // Loose gradient threshold

        let (mut state, first_point) = optimiser.init(vec![0.1, 0.1], Bounds::unbounded(2));
        let mut result = sphere_infallible(&first_point);

        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(results) => {
                    assert!(matches!(
                        results.termination,
                        TerminationReason::GradientToleranceReached
                    ));
                    break;
                }
            }
        }
    }

    #[test]
    fn adam_respects_bounds() {
        let optimiser = Adam::new().with_max_iter(100).with_step_size(1.0); // Large step size to test clamping

        let bounds = Bounds::new(vec![(-2.0, 2.0), (-2.0, 2.0)]);
        let (mut state, first_point) = optimiser.init(vec![1.5, -1.5], bounds.clone());
        let mut result = sphere_infallible(&first_point);

        // Verify initial point respects bounds
        assert!(first_point.iter().all(|&x| (-2.0..=2.0).contains(&x)));

        for _ in 0..20 {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Every point should respect bounds
                    for val in &points[0] {
                        assert!(
                            *val >= -2.0 && *val <= 2.0,
                            "Point {} violates bounds [-2.0, 2.0]",
                            val
                        );
                    }
                    result = sphere_infallible(&points[0]);
                }
                AskResult::Done(_) => break,
            }
        }
    }

    #[test]
    fn adam_rosenbrock_ask_tell() {
        let optimiser = Adam::new()
            .with_max_iter(5000)
            .with_step_size(0.001)
            .with_threshold(1e-6);

        let (mut state, first_point) = optimiser.init(vec![0.0, 0.0], Bounds::unbounded(2));
        let mut result = rosenbrock_infallible(&first_point);

        loop {
            state.tell(result).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    result = rosenbrock_infallible(&points[0]);
                }
                AskResult::Done(results) => {
                    // Rosenbrock is challenging - just verify we made progress
                    assert!(
                        results.value < 10.0,
                        "Should make progress on Rosenbrock, got {}",
                        results.value
                    );
                    break;
                }
            }
        }
    }

    #[test]
    fn adam_evaluation_error_propagation() {
        fn failing_function(_x: &[f64]) -> Result<(f64, Vec<f64>), std::io::Error> {
            Err(std::io::Error::other("Evaluation failed"))
        }

        let optimiser = Adam::new().with_max_iter(100).with_step_size(0.1);
        let (mut state, first_point) = optimiser.init(vec![1.0, 1.0], Bounds::unbounded(2));

        // First evaluation succeeds
        let result1 = sphere_fallible(&first_point).expect("first eval should succeed");
        state.tell(result1).expect("first tell should succeed");

        match state.ask() {
            AskResult::Evaluate(points) => {
                // Second evaluation fails
                let result2 = failing_function(&points[0]);
                state
                    .tell(result2)
                    .expect("tell should succeed even with error");

                // Should terminate due to evaluation failure
                match state.ask() {
                    AskResult::Done(results) => {
                        assert!(matches!(
                            results.termination,
                            TerminationReason::FunctionEvaluationFailed(_)
                        ));
                    }
                    _ => panic!("Should terminate after evaluation error"),
                }
            }
            AskResult::Done(_) => panic!("Should not terminate after first evaluation"),
        }
    }
}
