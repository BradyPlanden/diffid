use crate::optimisers::{
    build_results, ApplyBounds, AskResult, Bounds, EvaluatedPoint, Gradient, GradientEvaluation,
    GradientInput, IntoEvaluation, OptimisationResults, Point, TellError,
    TerminationReason,
};
use std::error::Error as StdError;
use std::time::{Duration, Instant};

/// Configuration for the Adam optimizer
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

    pub fn with_patience(mut self, patience: Duration) -> Self {
        self.patience = Some(patience);
        self
    }

    /// Get the effective gradient threshold
    fn gradient_threshold(&self) -> f64 {
        self.gradient_threshold.unwrap_or(self.threshold)
    }

    /// Initialize the optimization state
    ///
    /// Returns the state and the first point to evaluate
    pub fn init(&self, initial: Point, bounds: Option<Bounds>) -> (AdamState, Point) {
        let dim = initial.len();
        let mut initial_point = initial;
        initial_point.apply_bounds(bounds.as_ref());

        let state = AdamState::new(self.clone(), initial_point.clone(), bounds, dim);
        (state, initial_point)
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase Enum
// ─────────────────────────────────────────────────────────────────────────────

/// The current phase of the Adam algorithm
#[derive(Clone, Debug)]
pub enum AdamPhase {
    /// Waiting for point evaluation with gradient
    AwaitingEvaluation { pending_point: Point },

    /// Algorithm has terminated
    Terminated(TerminationReason),
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam Momentum State
// ─────────────────────────────────────────────────────────────────────────────

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
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            // Update biased second moment estimate
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

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

// ─────────────────────────────────────────────────────────────────────────────
// Adam State
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime state of the Adam optimizer
pub struct AdamState {
    config: Adam,
    bounds: Option<Bounds>,
    dim: usize,

    // Current position
    x: Point,

    // Adam momentum state
    momentum: MomentumState,

    // Current phase
    phase: AdamPhase,

    // Tracking
    nit: usize,
    nfev: usize,
    start_time: Instant,
    history: Vec<EvaluatedPoint>,
    prev_cost: Option<f64>,
}

impl AdamState {
    fn new(config: Adam, initial_point: Point, bounds: Option<Bounds>, dim: usize) -> Self {
        Self {
            config,
            bounds,
            dim,
            x: initial_point.clone(),
            momentum: MomentumState::new(dim),
            phase: AdamPhase::AwaitingEvaluation {
                pending_point: initial_point,
            },
            nit: 0,
            nfev: 0,
            start_time: Instant::now(),
            history: Vec::new(),
            prev_cost: None,
        }
    }

    /// Get the next point to evaluate, or the final result if optimization is complete
    pub fn ask(&self) -> AskResult {
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
    pub fn tell(&mut self, result: impl GradientInput) -> Result<(), TellError> {
        if matches!(self.phase, AdamPhase::Terminated(_)) {
            return Err(TellError::AlreadyTerminated);
        }

        // Convert to evaluation result
        let eval = match result.into_evaluation() {
            Ok(e) => e,
            Err(e) => {
                self.history
                    .push(EvaluatedPoint::new(self.x.clone(), f64::NAN));
                self.phase = AdamPhase::Terminated(TerminationReason::FunctionEvaluationFailed(
                    format!("{}", e),
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

        self.nfev += 1;

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
        self.nit
    }

    /// Get current function evaluation count
    pub fn evaluations(&self) -> usize {
        self.nfev
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

    // ─────────────────────────────────────────────────────────────────────────
    // Phase Handler
    // ─────────────────────────────────────────────────────────────────────────

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
        if self.nit >= self.config.max_iter {
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
        if let Some(ref bounds) = self.bounds {
            bounds.apply(&mut self.x);
        }

        self.nit += 1;

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
            self.nit,
            self.nfev,
            self.start_time.elapsed(),
            reason,
            None,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience wrapper
// ─────────────────────────────────────────────────────────────────────────────

impl Adam {
    /// Run optimization using a closure for evaluation
    ///
    /// The closure should return `(value, gradient)` for a given point
    pub fn run<F, R>(
        &self,
        mut objective: F,
        initial: Point,
        bounds: Option<Bounds>,
    ) -> OptimisationResults
    where
        F: FnMut(&[f64]) -> R,
        R: IntoEvaluation<GradientEvaluation>,
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

    /// Run optimization with numerical gradient approximation
    ///
    /// Uses central differences to approximate the gradient
    pub fn run_with_numerical_gradient<F, E>(
        &self,
        mut objective: F,
        initial: Point,
        bounds: Option<Bounds>,
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

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
        let (mut state, first_point) = adam.init(initial, None);
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
        let (mut state, first_point) = adam.init(initial, None);
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

        let results = adam.run(|x| sphere_infallible(x), vec![5.0, 5.0], None);

        println!("Final value: {}", results.value);
        println!("Iterations: {}", results.iterations);
        assert!(results.value < 1e-3);
    }

    #[test]
    fn test_numerical_gradient() {
        let adam = Adam::new().with_max_iter(500).with_step_size(0.1);

        let results = adam.run_with_numerical_gradient(
            |x| -> Result<f64, std::io::Error> { Ok(x.iter().map(|xi| xi * xi).sum()) },
            vec![2.0, 2.0],
            None,
            1e-5,
        );

        assert!(results.value < 1e-2);
    }

    #[test]
    fn test_with_bounds() {
        let adam = Adam::new().with_max_iter(500).with_step_size(0.1);

        let bounds = Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]);
        let results = adam.run(|x| sphere_infallible(x), vec![5.0, 5.0], Some(bounds));

        assert!(results.value < 1e-3);
    }

    #[test]
    fn test_rosenbrock() {
        let adam = Adam::new()
            .with_max_iter(10000)
            .with_step_size(0.001)
            .with_threshold(1e-8);

        let results = adam.run(|x| rosenbrock_infallible(x), vec![0.0, 0.0], None);

        println!("Rosenbrock result: {}", results.value);
        println!("Iterations: {}", results.iterations);
        // Rosenbrock is harder - just check we made progress
        assert!(results.value < 1.0);
    }

    #[test]
    fn test_gradient_dimension_mismatch() {
        let adam = Adam::new();
        let (mut state, _) = adam.init(vec![1.0, 2.0], None);

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
    fn test_gradient_tolerance() {
        let adam = Adam::new().with_gradient_threshold(1.0); // Very loose threshold

        let (mut state, first_point) = adam.init(vec![0.1, 0.1], None);

        // Start near origin with small gradient
        let val_grad = sphere_fallible(&first_point).expect("should be fine");
        state
            .tell(GradientEvaluation::new(val_grad.0, val_grad.1));

        match state.ask() {
            AskResult::Done(results) => {
                assert!(matches!(
                    results.termination,
                    TerminationReason::GradientToleranceReached
                ));
            }
            _ => panic!("Should have converged due to gradient tolerance"),
        }
    }
}
