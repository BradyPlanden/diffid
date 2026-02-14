use crate::common::{AskResult, Bounds, Point};
use crate::errors::{EvaluationError, TellError};
use crate::optimisers::{
    build_results, EvaluatedPoint, OptimisationResults, ScalarEvaluation, TerminationReason,
};
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the CMA-ES optimiser
#[must_use]
#[derive(Clone, Debug)]
pub struct CMAES {
    max_iter: usize,
    threshold: f64,
    step_size: f64,
    patience: Option<Duration>,
    population_size: Option<usize>,
    seed: Option<u64>,
}

impl CMAES {
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            threshold: 1e-6,
            step_size: 0.5,
            patience: None,
            population_size: None,
            seed: None,
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

    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size.max(1e-12);
        self
    }

    pub fn with_patience(mut self, patience: f64) -> Self {
        self.patience = Some(Duration::from_secs_f64(patience));
        self
    }

    pub fn with_population_size(mut self, population_size: usize) -> Self {
        if population_size >= 1 {
            self.population_size = Some(population_size);
        }
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn compute_population_size(&self, dim: usize) -> usize {
        self.population_size.unwrap_or_else(|| {
            if dim > 0 {
                let suggested = 3.0f64.mul_add((dim as f64).ln(), 4.0).floor() as usize;
                suggested.max(4).max(2 * dim)
            } else {
                4
            }
        })
    }

    /// Initialize the optimisation state
    ///
    /// Returns the state and the first point to evaluate
    pub fn init(&self, initial: Point, bounds: Bounds) -> (CMAESState, Point) {
        let dim = initial.len();
        let mut initial_point = initial;

        if let Err(err) = bounds.validate_dimension(dim) {
            let mut state = CMAESState::new(self.clone(), initial_point.clone(), bounds, dim);
            state.phase = CMAESPhase::Terminated(TerminationReason::FunctionEvaluationFailed(
                err.to_string(),
            ));
            return (state, initial_point);
        }

        bounds.clamp(&mut initial_point);

        let state = CMAESState::new(self.clone(), initial_point.clone(), bounds, dim);
        (state, initial_point)
    }
}

impl Default for CMAES {
    fn default() -> Self {
        Self::new()
    }
}

/// The current phase of the CMA-ES algorithm
#[derive(Clone, Debug)]
pub enum CMAESPhase {
    /// Waiting for initial point evaluation
    EvaluatingInitial { initial_point: Point },

    /// Waiting for population evaluation
    AwaitingPopulation {
        candidates: Arc<[Point]>,
        old_mean: DVector<f64>,
    },

    /// Algorithm has terminated
    Terminated(TerminationReason),
}

/// Pre-computed CMA-ES strategy parameters
#[derive(Clone, Debug)]
pub struct StrategyParameters {
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c1: f64,
    c_mu: f64,
    chi_n: f64,
}

impl StrategyParameters {
    fn new(dim: usize, lambda: usize) -> Self {
        let dim_f = dim as f64;

        let mu = (lambda / 2).max(1);

        // Compute weights
        let mut weights: Vec<f64> = (0..mu).map(|i| (mu - i) as f64).collect();
        let weight_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= weight_sum;
        }

        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let c_sigma = (mu_eff + 2.0) / (dim_f + mu_eff + 5.0);
        let d_sigma = Self::compute_d_sigma(mu_eff, dim_f, c_sigma);
        let c_c = (4.0 + mu_eff / dim_f) / (dim_f + 4.0 + 2.0 * mu_eff / dim_f);
        let c1 = 2.0 / (dim_f + 1.3).mul_add(dim_f + 1.3, mu_eff);
        let c_mu = ((1.0 - c1)
            .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (dim_f + 2.0).mul_add(dim_f + 2.0, mu_eff)))
        .max(0.0);
        let chi_n = dim_f.sqrt() * (1.0 - 1.0 / (4.0 * dim_f) + 1.0 / (21.0 * dim_f.powi(2)));

        Self {
            lambda,
            mu,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c1,
            c_mu,
            chi_n,
        }
    }

    pub fn compute_d_sigma(mu_eff: f64, dim_f: f64, c_sigma: f64) -> f64 {
        let sqrt_term = ((mu_eff - 1.0) / (dim_f + 1.0)).max(0.0).sqrt();
        2.0f64.mul_add((sqrt_term - 1.0).max(0.0), 1.0 + c_sigma)
    }
}

/// State of the CMA-ES sampling distribution
#[derive(Clone)]
struct DistributionState {
    mean: DVector<f64>,
    sigma: f64,
    cov: DMatrix<f64>,
    p_sigma: DVector<f64>,
    p_c: DVector<f64>,
    eigenvectors: DMatrix<f64>,
    sqrt_eigenvalues: DVector<f64>,
    inv_sqrt_cov: DMatrix<f64>,
    eigen_update_interval: usize,
    generations_since_eigen_update: usize,
    decomposition_stale: bool,
}

impl DistributionState {
    fn compute_eigen_update_interval(dim: usize) -> usize {
        (dim / 24).clamp(1, 8)
    }

    fn new(initial: &[f64]) -> Self {
        let dim = initial.len();
        Self {
            mean: DVector::from_column_slice(initial),
            sigma: 0.5, // Will be set by caller
            cov: DMatrix::identity(dim, dim),
            p_sigma: DVector::zeros(dim),
            p_c: DVector::zeros(dim),
            eigenvectors: DMatrix::identity(dim, dim),
            sqrt_eigenvalues: DVector::from_element(dim, 1.0),
            inv_sqrt_cov: DMatrix::identity(dim, dim),
            eigen_update_interval: Self::compute_eigen_update_interval(dim),
            generations_since_eigen_update: 0,
            decomposition_stale: false,
        }
    }

    fn update_decomposition(&mut self) {
        let dim = self.cov.nrows();
        if dim == 0 {
            return;
        }

        let sym = (&self.cov + self.cov.transpose()) * 0.5;
        let eig = sym.symmetric_eigen();
        self.eigenvectors = eig.eigenvectors;
        self.sqrt_eigenvalues = eig.eigenvalues.map(|val: f64| val.max(1e-30).sqrt());

        let inv_diag = self.sqrt_eigenvalues.map(|val| {
            if val > 0.0 {
                (1.0 / val).min(1e12_f64)
            } else {
                1e12
            }
        });
        self.inv_sqrt_cov =
            &self.eigenvectors * DMatrix::from_diagonal(&inv_diag) * self.eigenvectors.transpose();
        self.decomposition_stale = false;
        self.generations_since_eigen_update = 0;
    }

    fn mark_covariance_changed(&mut self) {
        self.decomposition_stale = true;
        self.generations_since_eigen_update = self.generations_since_eigen_update.saturating_add(1);
    }

    fn maybe_update_decomposition(&mut self) {
        if self.decomposition_stale
            && self.generations_since_eigen_update >= self.eigen_update_interval
        {
            self.update_decomposition();
        }
    }
}

#[derive(Clone)]
struct PopulationMember {
    evaluated: EvaluatedPoint,
    vector: DVector<f64>,
}

/// Runtime state of the CMA-ES optimiser
pub struct CMAESState {
    config: CMAES,
    bounds: Bounds,
    dim: usize,

    // CMA-ES internal state
    distribution: DistributionState,
    params: StrategyParameters,
    rng: StdRng,

    // Current phase
    phase: CMAESPhase,

    // Tracking
    iterations: usize,
    evaluations: usize,
    start_time: Instant,
    best_point: Option<EvaluatedPoint>,
    final_population: Vec<EvaluatedPoint>,
}

impl CMAESState {
    fn new(config: CMAES, initial_point: Point, bounds: Bounds, dim: usize) -> Self {
        let lambda = config.compute_population_size(dim);
        let params = StrategyParameters::new(dim, lambda);

        let mut distribution = DistributionState::new(&initial_point);
        distribution.sigma = config.step_size.max(1e-12);

        let rng = config
            .seed
            .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);

        Self {
            config,
            bounds,
            dim,
            distribution,
            params,
            rng,
            phase: CMAESPhase::EvaluatingInitial { initial_point },
            iterations: 0,
            evaluations: 0,
            start_time: Instant::now(),
            best_point: None,
            final_population: Vec::new(),
        }
    }

    /// Get the next point(s) to evaluate, or the final result if optimisation is complete
    pub fn ask(&self) -> AskResult<OptimisationResults> {
        match &self.phase {
            CMAESPhase::Terminated(reason) => AskResult::Done(self.build_results(reason.clone())),
            CMAESPhase::EvaluatingInitial { initial_point } => {
                AskResult::Evaluate(Arc::from(vec![initial_point.clone()]))
            }
            CMAESPhase::AwaitingPopulation { candidates, .. } => {
                AskResult::Evaluate(candidates.clone())
            }
        }
    }

    /// Report the evaluation results for the points from `ask()`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The optimiser has already terminated
    /// - The number of results doesn't match the expected population size
    pub fn tell<I, T, E>(&mut self, results: I) -> Result<(), TellError>
    where
        I: IntoIterator<Item = T>,
        T: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        if matches!(self.phase, CMAESPhase::Terminated(_)) {
            return Err(TellError::AlreadyTerminated);
        }

        // We collect into a Result<Vec<f64>> first to handle errors early
        let values: Vec<f64> = results
            .into_iter()
            .map(std::convert::TryInto::try_into)
            .map(|res| res.map_or(f64::INFINITY, |eval| eval.value()))
            .collect();

        // Take ownership of current phase
        // Placeholder MaxIters is set, will be replaced
        // in handle methods below
        let phase = std::mem::replace(
            &mut self.phase,
            CMAESPhase::Terminated(TerminationReason::MaxIterationsReached),
        );

        match phase {
            CMAESPhase::EvaluatingInitial { initial_point } => {
                self.handle_initial_evaluated(&initial_point, &values)?;
            }
            CMAESPhase::AwaitingPopulation {
                candidates,
                old_mean,
            } => {
                self.handle_population_evaluated(&candidates, &old_mean, values)?;
            }
            CMAESPhase::Terminated(_) => unreachable!(),
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
        self.best_point
            .as_ref()
            .map(|ep| (ep.point.as_slice(), ep.value))
    }

    /// Get current sigma (step size)
    pub fn sigma(&self) -> f64 {
        self.distribution.sigma
    }

    /// Get the current mean of the search distribution
    pub fn mean(&self) -> &[f64] {
        self.distribution.mean.as_slice()
    }

    /// Get reference to current covariance matrix
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.distribution.cov
    }

    // Phase Handlers
    fn handle_initial_evaluated(
        &mut self,
        initial_point: &Point,
        results: &[f64],
    ) -> Result<(), TellError> {
        if results.len() != 1 {
            return Err(TellError::ResultCountMismatch {
                expected: 1,
                got: results.len(),
            });
        }

        self.evaluations += 1;
        let value = results[0];

        let evaluated = EvaluatedPoint::new(initial_point.clone(), value);
        self.best_point = Some(evaluated.clone());
        self.final_population = vec![evaluated];

        if self.dim == 0 {
            self.phase = CMAESPhase::Terminated(TerminationReason::BothTolerancesReached);
            return Ok(());
        }

        self.start_generation();
        Ok(())
    }

    fn handle_population_evaluated(
        &mut self,
        candidates: &Arc<[Point]>,
        old_mean: &DVector<f64>,
        results: Vec<f64>,
    ) -> Result<(), TellError> {
        let expected = candidates.len();
        if results.len() != expected {
            return Err(TellError::ResultCountMismatch {
                expected,
                got: results.len(),
            });
        }

        self.evaluations += results.len();

        // Process results into population
        let mut population: Vec<PopulationMember> = Vec::with_capacity(expected);

        for (candidate, value) in candidates.iter().cloned().zip(results.into_iter()) {
            let vector = DVector::from_column_slice(&candidate);
            population.push(PopulationMember {
                evaluated: EvaluatedPoint::new(candidate, value),
                vector,
            });
        }

        // Sort by objective value
        population.sort_by(|a, b| {
            a.evaluated
                .value
                .partial_cmp(&b.evaluated.value)
                .unwrap_or(Ordering::Equal)
        });

        // Update best point
        if let Some(best) = population.first() {
            match &self.best_point {
                Some(current_best) if best.evaluated.value < current_best.value => {
                    self.best_point = Some(best.evaluated.clone());
                }
                None => {
                    self.best_point = Some(best.evaluated.clone());
                }
                _ => {}
            }
        }

        // Update CMA-ES state
        let termination_reason = self.update_distribution(&population, old_mean);

        // Update final population
        self.final_population = population
            .iter()
            .map(|member| member.evaluated.clone())
            .collect();
        if let Some(ref best) = self.best_point {
            if !self
                .final_population
                .iter()
                .any(|pt| pt.point == best.point)
            {
                self.final_population.push(best.clone());
            }
        }

        self.iterations += 1;

        // Check termination
        if let Some(reason) = termination_reason {
            self.phase = CMAESPhase::Terminated(reason);
            return Ok(());
        }

        // Start next generation
        self.start_generation();
        Ok(())
    }

    // State Transitions
    fn start_generation(&mut self) {
        // Check termination conditions first
        if let Some(reason) = self.check_pre_generation_termination() {
            self.phase = CMAESPhase::Terminated(reason);
            return;
        }

        // Update eigendecomposition
        self.distribution.maybe_update_decomposition();

        // Sample new population
        let old_mean = self.distribution.mean.clone();
        let candidates = self.sample_population();

        self.phase = CMAESPhase::AwaitingPopulation {
            candidates,
            old_mean,
        };
    }

    fn sample_population(&mut self) -> Arc<[Point]> {
        let lambda = self.params.lambda;

        let mut candidates: Vec<Point> = Vec::with_capacity(lambda);

        for _ in 0..lambda {
            let mut z = DVector::from_iterator(
                self.dim,
                (0..self.dim).map(|_| self.rng.sample::<f64, _>(StandardNormal)),
            );
            z.component_mul_assign(&self.distribution.sqrt_eigenvalues);
            let step = &self.distribution.eigenvectors * z;
            let mut candidate: Point = self
                .distribution
                .mean
                .iter()
                .zip(step.iter())
                .map(|(m, s)| self.distribution.sigma.mul_add(*s, *m))
                .collect();

            // Apply bounds
            self.bounds.clamp(&mut candidate);

            candidates.push(candidate);
        }

        Arc::from(candidates)
    }

    fn update_distribution(
        &mut self,
        population: &[PopulationMember],
        old_mean: &DVector<f64>,
    ) -> Option<TerminationReason> {
        let dim_f = self.dim as f64;
        let params = &self.params;
        let dist = &mut self.distribution;

        // Update mean
        let limit = params.mu.min(population.len());
        let mut new_mean = DVector::zeros(self.dim);
        for (i, item) in population.iter().enumerate().take(limit) {
            let weight = params.weights[i];
            new_mean += &item.vector * weight;
        }

        let mean_shift = &new_mean - old_mean;
        dist.mean = new_mean;

        // Update evolution paths
        let sigma_denom = dist.sigma.max(1e-12);
        let norm_factor = (params.c_sigma * (2.0 - params.c_sigma) * params.mu_eff).sqrt();
        let mean_shift_normalized = &mean_shift / sigma_denom;
        let delta = &dist.inv_sqrt_cov * &mean_shift_normalized;

        dist.p_sigma = &dist.p_sigma * (1.0 - params.c_sigma) + &delta * norm_factor;

        let norm_p_sigma = dist.p_sigma.norm();
        let exponent = 2.0 * ((self.iterations + 1) as f64);
        let factor = (1.0 - (1.0 - params.c_sigma).powf(exponent))
            .max(1e-12)
            .sqrt();
        let h_sigma_threshold = (1.4 + 2.0 / (dim_f + 1.0)) * params.chi_n;
        let h_sigma = if norm_p_sigma / factor < h_sigma_threshold {
            1.0
        } else {
            0.0
        };

        let pc_factor = (params.c_c * (2.0 - params.c_c) * params.mu_eff).sqrt();
        let mean_shift_scaled = &mean_shift * (h_sigma * pc_factor / sigma_denom);
        dist.p_c = &dist.p_c * (1.0 - params.c_c) + mean_shift_scaled;

        // Rank-mu update
        let mut rank_mu_update = DMatrix::zeros(self.dim, self.dim);
        for (i, item) in population.iter().enumerate().take(limit) {
            let weight = params.weights[i];
            let y = (&item.vector - old_mean) / sigma_denom;
            rank_mu_update += (&y * y.transpose()) * weight;
        }

        // Update covariance
        dist.cov = Self::update_covariance(
            &dist.cov,
            params.c1,
            params.c_mu,
            &dist.p_c,
            h_sigma,
            params.c_c,
            &rank_mu_update,
        );
        dist.mark_covariance_changed();

        // Update step size
        dist.sigma *= (params.c_sigma / params.d_sigma * (norm_p_sigma / params.chi_n - 1.0)).exp();
        dist.sigma = dist.sigma.max(1e-18);

        // Check convergence
        self.check_convergence(population, &mean_shift)
    }

    pub fn update_covariance(
        cov: &DMatrix<f64>,
        c1: f64,
        c_mu: f64,
        p_c: &DVector<f64>,
        h_sigma: f64,
        c_c: f64,
        rank_mu_contrib: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let mut updated = cov * (1.0 - c1 - c_mu);

        let correction_factor = (1.0 - h_sigma) * c_c * (2.0 - c_c);
        let rank_one = p_c * p_c.transpose();
        updated += rank_one * c1;
        updated += cov * (correction_factor * c1);
        updated += rank_mu_contrib * c_mu;

        updated
    }

    fn check_pre_generation_termination(&self) -> Option<TerminationReason> {
        // Check patience/timeout
        if let Some(patience) = self.config.patience {
            if self.start_time.elapsed() >= patience {
                return Some(TerminationReason::PatienceElapsed);
            }
        }

        // Check max iterations
        if self.iterations >= self.config.max_iter {
            return Some(TerminationReason::MaxIterationsReached);
        }

        None
    }

    fn check_convergence(
        &self,
        population: &[PopulationMember],
        mean_shift: &DVector<f64>,
    ) -> Option<TerminationReason> {
        let fun_diff = if population.len() > 1 {
            let best_val = population[0].evaluated.value;
            let worst_val = population[population.len() - 1].evaluated.value;
            (worst_val - best_val).abs()
        } else {
            0.0
        };

        let position_converged = mean_shift.norm() <= self.config.threshold;
        let fun_converged = fun_diff <= self.config.threshold;

        match (fun_converged, position_converged) {
            (true, true) => Some(TerminationReason::BothTolerancesReached),
            (true, false) => Some(TerminationReason::FunctionToleranceReached),
            (false, true) => Some(TerminationReason::ParameterToleranceReached),
            (false, false) => None,
        }
    }

    fn build_results(&self, reason: TerminationReason) -> OptimisationResults {
        build_results(
            &self.final_population,
            self.iterations,
            self.evaluations,
            self.start_time.elapsed(),
            reason,
            Some(&self.distribution.cov),
        )
    }
}

impl CMAES {
    /// Run optimisation using a closure for evaluation
    ///
    /// This is a convenience wrapper around the ask/tell interface
    ///
    /// # Panics
    ///
    /// Panics if the optimisation state machine enters an unexpected state after
    /// a tell error. This should not occur under normal operation.
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
        if let Err(err) = bounds.validate_dimension(initial.len()) {
            return build_results(
                &[EvaluatedPoint::new(initial, f64::NAN)],
                0,
                0,
                Duration::ZERO,
                TerminationReason::FunctionEvaluationFailed(err.to_string()),
                None,
            );
        }

        let (mut state, first_point) = self.init(initial, bounds);

        let mut results = vec![objective(&first_point)];

        loop {
            if state.tell(results).is_err() {
                break;
            }

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| objective(p)).collect();
                }
                AskResult::Done(opt_results) => {
                    return opt_results;
                }
            }
        }

        match state.ask() {
            AskResult::Done(opt_results) => opt_results,
            AskResult::Evaluate(_) => panic!("Unexpected state after tell error"),
        }
    }

    /// Run optimisation using batch evaluation
    ///
    /// Similar to `run`, but evaluates all population points in a single call
    ///
    /// # Panics
    ///
    /// Panics if the optimisation state machine enters an unexpected state after
    /// a tell error. This should not occur under normal operation.
    pub fn run_batch<F, R, E>(
        &self,
        objective: F,
        initial: Point,
        bounds: Bounds,
    ) -> OptimisationResults
    where
        F: Fn(&[Vec<f64>]) -> Vec<R>,
        R: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        if let Err(err) = bounds.validate_dimension(initial.len()) {
            return build_results(
                &[EvaluatedPoint::new(initial, f64::NAN)],
                0,
                0,
                Duration::ZERO,
                TerminationReason::FunctionEvaluationFailed(err.to_string()),
                None,
            );
        }

        let (mut state, first_point) = self.init(initial, bounds);

        let mut results = objective(&[first_point]);

        loop {
            if state.tell(results).is_err() {
                break;
            }

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = objective(points.as_ref());
                }
                AskResult::Done(opt_results) => {
                    return opt_results;
                }
            }
        }

        match state.ask() {
            AskResult::Done(opt_results) => opt_results,
            AskResult::Evaluate(_) => panic!("Unexpected state after tell error"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::ScalarProblemBuilder;
    use std::convert::Infallible;

    #[allow(clippy::unnecessary_wraps)]
    fn sphere(x: &[f64]) -> Result<f64, Infallible> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    #[test]
    fn test_ask_tell_interface() {
        let cmaes = CMAES::new().with_max_iter(100).with_seed(42);

        let initial = vec![5.0, 5.0];
        let (mut state, first_point) = cmaes.init(initial, Bounds::unbounded(2));

        let mut current_results = vec![sphere(&first_point)];

        loop {
            match state.tell(current_results) {
                Ok(()) => {}
                Err(TellError::AlreadyTerminated) => break,
                Err(e) => panic!("Unexpected error: {e:?}"),
            }

            match state.ask() {
                AskResult::Evaluate(points) => {
                    current_results = points.iter().map(|p| sphere(p)).collect();
                }
                AskResult::Done(results) => {
                    println!("optimisation complete!");
                    println!("Best value: {}", results.value);
                    println!("Iterations: {}", results.iterations);
                    println!("Evaluations: {}", results.evaluations);
                    assert!(results.value < 1e-3);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_run_convenience_wrapper() {
        let cmaes = CMAES::new().with_max_iter(100).with_seed(42);

        let results = cmaes.run(sphere, vec![5.0, 5.0], Bounds::unbounded(2));

        assert!(results.value < 1e-3);
    }

    #[test]
    fn test_with_bounds() {
        let cmaes = CMAES::new().with_max_iter(100).with_seed(42);

        // Todo: convert run to accept vec!, then convert to Bounds if needed
        let bounds = Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]);
        let results = cmaes.run(sphere, vec![5.0, 5.0], bounds);

        assert!(results.value < 1e-3);
    }

    #[test]
    fn cmaes_lazy_eigendecomposition_works() {
        // Test with high dimension to trigger lazy updates
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
            .build()
            .unwrap();

        let dim = 60; // > 50 to trigger lazy updates
        let initial = vec![0.5; dim];

        let optimiser = CMAES::new()
            .with_max_iter(100)
            .with_threshold(1e-6)
            .with_seed(777);

        let initial_value = initial.iter().map(|x| x * x).sum::<f64>();

        let result = optimiser.run(|x| problem.evaluate(x), initial, Bounds::unbounded(dim));

        // Should still work with lazy updates and improve from initial
        assert!(result.evaluations > 0);
        assert!(
            result.value < initial_value,
            "Should improve: {} < {}",
            result.value,
            initial_value
        );
    }

    #[test]
    fn cmaes_covariance_is_symmetric_and_psd() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
            .build()
            .unwrap();

        let optimiser = CMAES::new()
            .with_max_iter(400)
            .with_threshold(1e-10)
            .with_step_size(0.6)
            .with_seed(4242);

        let result = optimiser.run(
            |x| problem.evaluate(x),
            vec![4.5, -3.5],
            Bounds::unbounded(2),
        );

        assert!(result.success, "Expected success: {}", result.message);
        assert!(
            result.value < 1e-6,
            "Should reach low objective value: {}",
            result.value
        );

        let covariance = result
            .covariance
            .clone()
            .expect("CMAES should provide covariance estimates");
        assert_eq!(covariance.len(), 2);
        assert!(covariance.iter().all(|row| row.len() == 2));

        covariance
            .iter()
            .zip(&covariance)
            .for_each(|(row_i, row_j)| {
                row_i.iter().zip(row_j).for_each(|(a, b)| {
                    assert!((a - b).abs() < 1e-12, "covariance matrix must be symmetric");
                });
            });

        let flat: Vec<f64> = covariance
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let matrix = DMatrix::from_row_slice(2, 2, &flat);
        let eigenvalues = matrix.symmetric_eigen().eigenvalues;

        assert!(
            eigenvalues.iter().all(|&eig| eig >= -1e-10),
            "covariance must be positive semi-definite: {eigenvalues:?}"
        );
    }

    #[test]
    fn d_sigma_matches_hansen_2016_formula() {
        // Test case 1: Standard parameters from 10-dimensional optimisation
        let mu_eff = 4.5;
        let dim_f = 10.0;
        let c_sigma = 0.3;

        // Manual computation following Hansen (2016)
        let inner: f64 = (mu_eff - 1.0) / (dim_f + 1.0); // (4.5 - 1) / 11 = 0.318...
        let sqrt_inner = inner.sqrt(); // ~0.564
        let clamped = (sqrt_inner - 1.0).max(0.0); // max(0, -0.436) = 0.0
        let expected = 1.0 + c_sigma + 2.0 * clamped; // 1.0 + 0.3 + 0 = 1.3

        let computed = StrategyParameters::compute_d_sigma(mu_eff, dim_f, c_sigma);

        assert!(
            (computed - expected).abs() < 1e-12,
            "d_sigma mismatch: expected {expected}, got {computed}"
        );

        // For this case, the sqrt term is less than 1, so it should clamp to 0
        assert!(
            (computed - (1.0 + c_sigma)).abs() < 1e-12,
            "When sqrt((mu_eff-1)/(n+1)) < 1, d_sigma should equal 1 + c_sigma"
        );
    }

    #[test]
    fn cmaes_d_sigma_clamps_when_below_unity() {
        let mu_eff = 2.0_f64;
        let dim_f = 10.0_f64;
        let c_sigma = 0.2_f64;

        let expected = 1.0 + c_sigma;
        let computed = StrategyParameters::compute_d_sigma(mu_eff, dim_f, c_sigma);

        assert!((computed - expected).abs() < 1e-12);
    }

    #[test]
    fn covariance_update_applies_exponential_correction() {
        let cov = DMatrix::from_row_slice(2, 2, &[2.0, 0.1, 0.1, 1.0]);
        let c1 = 0.3_f64;
        let c_mu = 0.2_f64;
        let c_c = 0.5_f64;
        let h_sigma = 0.0_f64;
        let p_c = DVector::from_vec(vec![1.0, -0.5]);
        let rank_mu = DMatrix::zeros(2, 2);

        let correction_factor = (1.0 - h_sigma) * c_c * (2.0 - c_c);
        let expected = cov.clone() * (1.0 - c1 - c_mu)
            + (p_c.clone() * p_c.transpose() + cov.clone() * correction_factor) * c1
            + rank_mu.clone() * c_mu;

        let updated = CMAESState::update_covariance(&cov, c1, c_mu, &p_c, h_sigma, c_c, &rank_mu);

        for (exp, got) in expected.iter().zip(updated.iter()) {
            assert!((exp - got).abs() < 1e-12, "expected {exp} got {got}");
        }
    }

    #[test]
    fn covariance_update_skips_correction_when_h_sigma_one() {
        let cov = DMatrix::from_row_slice(2, 2, &[1.5, 0.2, 0.2, 0.8]);
        let c1 = 0.25_f64;
        let c_mu = 0.1_f64;
        let c_c = 0.6_f64;
        let h_sigma = 1.0_f64;
        let p_c = DVector::from_vec(vec![0.3, -0.7]);
        let rank_mu = DMatrix::from_row_slice(2, 2, &[0.05, 0.01, 0.01, 0.04]);

        let correction_factor = (1.0 - h_sigma) * c_c * (2.0 - c_c);
        let expected = cov.clone() * (1.0 - c1 - c_mu)
            + (p_c.clone() * p_c.transpose() + cov.clone() * correction_factor) * c1
            + rank_mu.clone() * c_mu;

        let updated = CMAESState::update_covariance(&cov, c1, c_mu, &p_c, h_sigma, c_c, &rank_mu);

        for (exp, got) in expected.iter().zip(updated.iter()) {
            assert!((exp - got).abs() < 1e-12, "expected {exp} got {got}");
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn sphere_cmaes(x: &[f64]) -> Result<f64, std::io::Error> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    #[test]
    fn cmaes_ask_tell_basic_workflow() {
        let optimiser = CMAES::new()
            .with_max_iter(100)
            .with_threshold(1e-6)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![3.0, -2.0], Bounds::unbounded(2));

        // CMAES asks for population of points
        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];

        loop {
            state.tell(results).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    assert!(final_results.value < 1e-4, "Should converge to minimum");
                    assert!(final_results.x[0].abs() < 0.1, "x[0] should be near 0");
                    assert!(final_results.x[1].abs() < 0.1, "x[1] should be near 0");
                    assert!(final_results.success, "Should report success");
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_ask_tell_matches_run() {
        let optimiser = CMAES::new()
            .with_max_iter(80)
            .with_threshold(1e-6)
            .with_seed(123);

        // Ask-tell version
        let (mut state, first_point) = optimiser.init(vec![5.0, -5.0], Bounds::unbounded(2));

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];
        let ask_tell_results = loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => break final_results,
            }
        };

        // .run() version
        let run_results = optimiser.run(sphere_cmaes, vec![5.0, -5.0], Bounds::unbounded(2));

        // Should produce identical results (deterministic with same seed)
        assert_eq!(ask_tell_results.iterations, run_results.iterations);
        assert!((ask_tell_results.value - run_results.value).abs() < 1e-12);
        assert_eq!(ask_tell_results.evaluations, run_results.evaluations);
    }

    #[test]
    fn cmaes_population_evaluation_batching() {
        let optimiser = CMAES::new().with_population_size(10).with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![1.0, 1.0], Bounds::unbounded(2));

        // Evaluate initial point
        let mut results = vec![sphere_cmaes(&first_point)];
        state.tell(results).unwrap();

        // First ask should request population_size points
        match state.ask() {
            AskResult::Evaluate(points) => {
                assert_eq!(points.len(), 10, "Should request population_size points");
                results = points.iter().map(|p| sphere_cmaes(p)).collect();
                state.tell(results).unwrap();
            }
            AskResult::Done(_) => panic!("Should not terminate on first ask"),
        }

        // Subsequent asks should also request population_size points
        match state.ask() {
            AskResult::Evaluate(points) => {
                assert_eq!(
                    points.len(),
                    10,
                    "Should consistently request population_size points"
                );
            }
            AskResult::Done(_) => panic!("Should not terminate immediately"),
        }
    }

    #[test]
    fn cmaes_tell_already_terminated() {
        let optimiser = CMAES::new()
            .with_max_iter(1)
            .with_threshold(1e-10)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![0.0], Bounds::unbounded(1));

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];
        loop {
            state.tell(results).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(_) => break,
            }
        }

        // Now try to tell again after termination
        let err = state
            .tell(
                (0..5)
                    .map(|_| Ok::<f64, std::io::Error>(1.0))
                    .collect::<Vec<_>>(),
            )
            .unwrap_err();
        assert!(matches!(err, TellError::AlreadyTerminated));
    }

    #[test]
    fn cmaes_result_count_mismatch() {
        let optimiser = CMAES::new().with_population_size(8).with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![1.0], Bounds::unbounded(1));

        // Evaluate initial point
        state.tell(vec![sphere_cmaes(&first_point)]).unwrap();

        // Get first ask which should request 8 points (population_size)
        match state.ask() {
            AskResult::Evaluate(points) => {
                assert_eq!(points.len(), 8, "Should request 8 points");

                // Provide wrong number of results (should be 8, provide 5)
                let wrong_results: Vec<Result<f64, std::io::Error>> =
                    (0..5).map(|_| Ok(1.0)).collect();
                let err = state.tell(wrong_results).unwrap_err();
                assert!(matches!(
                    err,
                    TellError::ResultCountMismatch {
                        expected: 8,
                        got: 5
                    }
                ));
            }
            AskResult::Done(_) => panic!("Should not be done yet"),
        }
    }

    #[test]
    fn cmaes_handles_evaluation_errors() {
        let optimiser = CMAES::new()
            .with_max_iter(20)
            .with_population_size(6)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![2.0, 2.0], Bounds::unbounded(2));

        let mut generation = 0;
        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];

        loop {
            state.tell(results).expect("tell should succeed");

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Inject errors in some population members on generation 3
                    results = points
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            if generation == 3 && i < 2 {
                                Err(std::io::Error::other("Simulated failure"))
                            } else {
                                sphere_cmaes(&p[..])
                            }
                        })
                        .collect();
                    generation += 1;
                }
                AskResult::Done(final_results) => {
                    // Should complete despite errors
                    assert!(final_results.iterations > 3, "Should continue past error");
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_respects_bounds_ask_tell() {
        let optimiser = CMAES::new().with_max_iter(50).with_seed(42);

        let bounds = Bounds::new(vec![(-2.0, 2.0), (-2.0, 2.0)]);
        let (mut state, first_point) = optimiser.init(vec![1.5, 1.5], bounds);

        // Check initial point respects bounds
        for &val in &first_point[..] {
            assert!(
                (-2.0..=2.0).contains(&val),
                "Initial point {first_point:?} violates bounds"
            );
        }

        let mut results = vec![sphere_cmaes(&first_point)];

        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Check all proposed points respect bounds
                    for point in points.iter() {
                        for &val in point {
                            assert!(
                                (-2.0..=2.0).contains(&val),
                                "Point {point:?} violates bounds"
                            );
                        }
                    }
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    // Final result should also respect bounds
                    for &val in &final_results.x {
                        assert!((-2.0..=2.0).contains(&val), "Final result violates bounds");
                    }
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_convergence_by_threshold() {
        let optimiser = CMAES::new()
            .with_max_iter(200)
            .with_threshold(1e-6)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![4.0, -3.0], Bounds::unbounded(2));

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];
        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    // Should converge by threshold, not max iterations
                    assert!(
                        final_results.iterations < 200,
                        "Should converge before max iterations"
                    );
                    assert!(
                        final_results.value < 1e-6,
                        "Should meet threshold requirement"
                    );
                    assert!(final_results.success, "Should report success");
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_termination_by_max_iter() {
        let optimiser = CMAES::new()
            .with_max_iter(5)
            .with_threshold(1e-12) // Very strict threshold
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![8.0], Bounds::unbounded(1));

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];
        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    assert_eq!(
                        final_results.iterations, 5,
                        "Should terminate at max iterations"
                    );
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_covariance_evolution() {
        let optimiser = CMAES::new().with_max_iter(30).with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![5.0, 5.0], Bounds::unbounded(2));

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];
        let mut iterations = 0;

        loop {
            state.tell(results).unwrap();

            iterations += 1;

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Verify population size is consistent
                    assert!(!points.is_empty(), "Should have population");
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    // Should have done multiple iterations to evolve covariance
                    assert!(iterations > 5, "Should run multiple iterations");
                    assert!(
                        final_results.evaluations > iterations,
                        "Evaluations should exceed iterations"
                    );
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_sigma_adaptation() {
        let optimiser = CMAES::new()
            .with_max_iter(50)
            .with_step_size(0.5)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![3.0], Bounds::unbounded(1));

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];

        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Sigma should adapt over time; verify points are being generated
                    assert!(!points.is_empty(), "Should generate points");
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    // Should converge despite starting sigma
                    assert!(final_results.value < 1.0, "Should improve significantly");
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_multidimensional() {
        let optimiser = CMAES::new()
            .with_max_iter(150)
            .with_threshold(1e-5)
            .with_seed(777);

        // 6D sphere function
        let (mut state, first_point) =
            optimiser.init(vec![3.0, -2.0, 1.5, -1.0, 2.5, -3.5], Bounds::unbounded(6));

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];

        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    assert_eq!(points[0].len(), 6, "Should maintain dimensionality");
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    assert_eq!(final_results.x.len(), 6, "Result should have 6 dimensions");
                    assert!(final_results.value < 1e-3, "Should converge to minimum");
                    for &x in &final_results.x {
                        assert!(x.abs() < 0.5, "Each coordinate should be near 0");
                    }
                    break;
                }
            }
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn rosenbrock_cmaes(x: &[f64]) -> Result<f64, std::io::Error> {
        let a = 1.0;
        let b = 100.0;
        let x0 = x[0];
        let x1 = x[1];
        Ok((a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2))
    }

    #[test]
    fn cmaes_rosenbrock_ask_tell() {
        let optimiser = CMAES::new()
            .with_max_iter(300)
            .with_threshold(1e-5)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![0.0, 0.0], Bounds::unbounded(2));

        let mut results: Vec<_> = vec![rosenbrock_cmaes(&first_point)];

        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    results = points.iter().map(|p| rosenbrock_cmaes(p)).collect();
                }
                AskResult::Done(final_results) => {
                    assert!(final_results.value < 1e-3, "Should converge on Rosenbrock");
                    assert!(
                        (final_results.x[0] - 1.0).abs() < 0.1,
                        "x[0] should be near 1"
                    );
                    assert!(
                        (final_results.x[1] - 1.0).abs() < 0.1,
                        "x[1] should be near 1"
                    );
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_state_machine_integrity() {
        let optimiser = CMAES::new()
            .with_max_iter(15)
            .with_population_size(6)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![1.0], Bounds::unbounded(1));

        let mut ask_count = 0;
        let mut tell_count = 1; // Initial tell about to happen

        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];

        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    ask_count += 1;
                    results = points.iter().map(|p| sphere_cmaes(p)).collect();
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

    #[test]
    fn cmaes_nonfinite_handling() {
        let optimiser = CMAES::new()
            .with_max_iter(30)
            .with_population_size(5)
            .with_seed(42);

        let (mut state, first_point) = optimiser.init(vec![2.0], Bounds::unbounded(1));

        let mut generation = 0;
        let mut results: Vec<_> = vec![sphere_cmaes(&first_point)];

        loop {
            state.tell(results).unwrap();

            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Return NaN for one member on generation 5
                    results = points
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            if generation == 5 && i == 0 {
                                Ok(f64::NAN)
                            } else {
                                sphere_cmaes(&p[..])
                            }
                        })
                        .collect();
                    generation += 1;
                }
                AskResult::Done(final_results) => {
                    // Should handle NaN and continue
                    assert!(
                        final_results.value.is_finite(),
                        "Final value should be finite"
                    );
                    assert!(generation > 5, "Should continue past NaN");
                    break;
                }
            }
        }
    }

    #[test]
    fn cmaes_reproducibility_with_seed() {
        let seed = 999;

        let optimiser1 = CMAES::new().with_max_iter(50).with_seed(seed);

        let (mut state1, first_point1) = optimiser1.init(vec![2.0, -1.0], Bounds::unbounded(2));
        let mut results1 = vec![sphere_cmaes(&first_point1)];

        let result1 = loop {
            state1.tell(results1).unwrap();
            match state1.ask() {
                AskResult::Evaluate(points) => {
                    results1 = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(r) => break r,
            }
        };

        let optimiser2 = CMAES::new().with_max_iter(50).with_seed(seed);

        let (mut state2, first_point2) = optimiser2.init(vec![2.0, -1.0], Bounds::unbounded(2));
        let mut results2 = vec![sphere_cmaes(&first_point2)];

        let result2 = loop {
            state2.tell(results2).unwrap();
            match state2.ask() {
                AskResult::Evaluate(points) => {
                    results2 = points.iter().map(|p| sphere_cmaes(p)).collect();
                }
                AskResult::Done(r) => break r,
            }
        };

        // With same seed, should produce identical results
        assert_eq!(result1.iterations, result2.iterations);
        assert_eq!(result1.evaluations, result2.evaluations);
        assert!((result1.value - result2.value).abs() < 1e-12);
        for i in 0..result1.x.len() {
            assert!((result1.x[i] - result2.x[i]).abs() < 1e-12);
        }
    }
}
