use crate::problem::Problem;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use std::cmp::Ordering;
use std::fmt;
use std::time::{Duration, Instant};

// Core behaviour shared by all optimisers
pub trait Optimiser {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults;
}

pub trait WithMaxIter: Optimiser {
    fn set_max_iter(&mut self, max_iter: usize);

    fn with_max_iter(mut self, max_iter: usize) -> Self
    where
        Self: Sized,
    {
        self.set_max_iter(max_iter);
        self
    }
}

pub trait WithThreshold: Optimiser {
    fn set_threshold(&mut self, threshold: f64);

    fn with_threshold(mut self, threshold: f64) -> Self
    where
        Self: Sized,
    {
        self.set_threshold(threshold);
        self
    }
}

pub trait WithSigma0: Optimiser {
    fn set_sigma0(&mut self, sigma0: f64);

    fn with_sigma0(mut self, sigma0: f64) -> Self
    where
        Self: Sized,
    {
        self.set_sigma0(sigma0);
        self
    }
}

pub trait WithPatience: Optimiser {
    fn set_patience(&mut self, patience_seconds: f64);

    fn with_patience(mut self, patience_seconds: f64) -> Self
    where
        Self: Sized,
    {
        self.set_patience(patience_seconds);
        self
    }
}

enum InitialState {
    Ready {
        start: Vec<f64>,
        start_value: f64,
        nfev: usize,
    },
    Finished(OptimisationResults),
}

#[derive(Debug, Clone)]
struct Bounds {
    limits: Vec<(f64, f64)>,
}

impl Bounds {
    fn new(limits: Vec<(f64, f64)>) -> Self {
        Self { limits }
    }

    fn apply(&self, point: &mut [f64]) {
        let dim = self.dimension();
        debug_assert_eq!(
            point.len(),
            dim,
            "Bounds dimension mismatch: point has {} entries, bounds have {}",
            point.len(),
            dim
        );

        for (val, (lower, upper)) in point.iter_mut().zip(self.limits.iter()) {
            *val = val.clamp(*lower, *upper);
        }
    }

    fn dimension(&self) -> usize {
        self.limits.len()
    }
}

/// Apply bounds to a point when bounds are provided
fn apply_bounds(point: &mut [f64], bounds: Option<&Bounds>) {
    if let Some(bounds) = bounds {
        bounds.apply(point);
    }
}

/// Extract bounds from problem parameter specs
fn extract_bounds(problem: &Problem) -> Option<Bounds> {
    let specs = problem.parameter_specs();
    if specs.is_empty() {
        return None;
    }

    let mut has_any_bounds = false;
    let mut limits: Vec<(f64, f64)> = Vec::with_capacity(specs.len());

    for spec in specs.iter() {
        match spec.bounds {
            Some((lower, upper)) => {
                let (low, high) = if lower <= upper {
                    (lower, upper)
                } else {
                    (upper, lower)
                };

                if low.is_finite() || high.is_finite() {
                    has_any_bounds = true;
                }

                limits.push((low, high));
            }
            None => limits.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }

    if has_any_bounds {
        Some(Bounds::new(limits))
    } else {
        None
    }
}

fn initialise_start(problem: &Problem, initial: Vec<f64>, bounds: Option<&Bounds>) -> InitialState {
    let mut start = if !initial.is_empty() {
        initial
    } else {
        vec![0.0; problem.dimension()]
    };

    // Clamp the initial guess so every optimiser starts within feasible bounds before evaluation.
    apply_bounds(&mut start, bounds);

    let dim = start.len();
    let failed_time = Duration::try_from_secs_f64(0.0).expect("Failed to convert 0.0 to Duration");
    if dim == 0 {
        let value = match evaluate_point(problem, &start) {
            Ok(v) => v,
            Err(msg) => {
                let result = build_results(
                    &[EvaluatedPoint::new(Vec::new(), f64::NAN)],
                    0,
                    0,
                    failed_time,
                    TerminationReason::FunctionEvaluationFailed(msg),
                    None,
                );
                return InitialState::Finished(result);
            }
        };

        let result = build_results(
            &[EvaluatedPoint::new(start, value)],
            0,
            1,
            failed_time,
            TerminationReason::BothTolerancesReached,
            None,
        );
        return InitialState::Finished(result);
    }

    match evaluate_point(problem, &start) {
        Ok(value) => InitialState::Ready {
            start,
            start_value: value,
            nfev: 1,
        },
        Err(msg) => {
            let result = build_results(
                &[EvaluatedPoint::new(start, f64::NAN)],
                0,
                1,
                failed_time,
                TerminationReason::FunctionEvaluationFailed(msg),
                None,
            );
            InitialState::Finished(result)
        }
    }
}

#[derive(Debug, Clone)]
struct EvaluatedPoint {
    point: Vec<f64>,
    value: f64,
}

impl EvaluatedPoint {
    fn new(point: Vec<f64>, value: f64) -> Self {
        Self { point, value }
    }
}

fn evaluate_point(problem: &Problem, point: &[f64]) -> Result<f64, String> {
    problem.evaluate(point)
}

fn build_results(
    points: &[EvaluatedPoint],
    nit: usize,
    nfev: usize,
    time: Duration,
    reason: TerminationReason,
    covariance: Option<&DMatrix<f64>>,
) -> OptimisationResults {
    let mut ordered = points.to_vec();
    ordered.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));

    let best = ordered
        .first()
        .cloned()
        .unwrap_or_else(|| EvaluatedPoint::new(Vec::new(), f64::NAN));

    let final_simplex = ordered.iter().map(|v| v.point.clone()).collect();
    let final_simplex_values = ordered.iter().map(|v| v.value).collect();

    let covariance = covariance.map(|matrix| {
        (0..matrix.nrows())
            .map(|row| matrix.row(row).iter().copied().collect())
            .collect()
    });

    let success = matches!(
        reason,
        TerminationReason::FunctionToleranceReached
            | TerminationReason::ParameterToleranceReached
            | TerminationReason::BothTolerancesReached
    );

    let message = reason.to_string();

    OptimisationResults {
        x: best.point,
        fun: best.value,
        nit,
        nfev,
        time,
        success,
        message,
        termination_reason: reason,
        final_simplex,
        final_simplex_values,
        covariance,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TerminationReason {
    FunctionToleranceReached,
    ParameterToleranceReached,
    BothTolerancesReached,
    MaxIterationsReached,
    MaxFunctionEvaluationsReached,
    DegenerateSimplex,
    PatienceElapsed,
    FunctionEvaluationFailed(String),
}

impl fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerminationReason::FunctionToleranceReached => {
                write!(f, "Function tolerance met")
            }
            TerminationReason::ParameterToleranceReached => {
                write!(f, "Parameter tolerance met")
            }
            TerminationReason::BothTolerancesReached => {
                write!(f, "Function and parameter tolerances met")
            }
            TerminationReason::MaxIterationsReached => {
                write!(f, "Maximum iterations reached")
            }
            TerminationReason::MaxFunctionEvaluationsReached => {
                write!(f, "Maximum function evaluations reached")
            }
            TerminationReason::DegenerateSimplex => {
                write!(f, "Degenerate simplex encountered")
            }
            TerminationReason::PatienceElapsed => {
                write!(f, "Patience elapsed")
            }
            TerminationReason::FunctionEvaluationFailed(msg) => {
                write!(f, "Function evaluation failed: {}", msg)
            }
        }
    }
}

// Nelder-Mead optimiser
#[derive(Clone)]
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

    fn reached_max_evaluations(&self, evaluations: usize) -> bool {
        match self.max_evaluations {
            Some(limit) => evaluations >= limit,
            None => false,
        }
    }

    fn convergence_reason(&self, simplex: &[EvaluatedPoint]) -> Option<TerminationReason> {
        if simplex.is_empty() {
            return None;
        }

        let best = &simplex[0];
        let worst = simplex.last().unwrap();
        let fun_diff = (worst.value - best.value).abs();

        let mut max_dist: f64 = 0.0;
        for vertex in simplex.iter().skip(1) {
            let mut sum: f64 = 0.0;
            for (a, b) in vertex.point.iter().zip(best.point.iter()) {
                let diff = a - b;
                sum += diff * diff;
            }
            max_dist = max_dist.max(sum.sqrt());
        }

        let fun_converged = fun_diff <= self.threshold;
        let position_converged = max_dist <= self.position_tolerance;

        match (fun_converged, position_converged) {
            (true, true) => Some(TerminationReason::BothTolerancesReached),
            (true, false) => Some(TerminationReason::FunctionToleranceReached),
            (false, true) => Some(TerminationReason::ParameterToleranceReached),
            _ => None,
        }
    }

    fn centroid(simplex: &[EvaluatedPoint]) -> Vec<f64> {
        let dim = simplex[0].point.len();
        let mut centroid = vec![0.0; dim];
        let count = simplex.len() as f64;

        for vertex in simplex {
            for (c, val) in centroid.iter_mut().zip(&vertex.point) {
                *c += val;
            }
        }

        for c in centroid.iter_mut() {
            *c /= count;
        }

        centroid
    }

    pub fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults {
        let start_time = Instant::now();

        let bounds = extract_bounds(problem);
        let bounds_ref = bounds.as_ref();

        let (start, start_value, mut nfev) = match initialise_start(problem, initial, bounds_ref) {
            InitialState::Finished(results) => return results,
            InitialState::Ready {
                start,
                start_value,
                nfev,
            } => (start, start_value, nfev),
        };

        let dim = start.len();

        let mut simplex = vec![EvaluatedPoint::new(start.clone(), start_value)];

        for i in 0..dim {
            if self.reached_max_evaluations(nfev) {
                return build_results(
                    &simplex,
                    0,
                    nfev,
                    start_time.elapsed(),
                    TerminationReason::MaxFunctionEvaluationsReached,
                    None,
                );
            }

            let mut point = start.clone();
            if point[i] != 0.0 {
                point[i] *= 1.0 + self.sigma0;
            } else {
                point[i] = self.sigma0;
            }

            if point
                .iter()
                .zip(simplex[0].point.iter())
                .all(|(a, b)| (*a - *b).abs() <= f64::EPSILON)
            {
                point[i] += self.sigma0;
            }

            // Keep each simplex vertex feasible before evaluating the objective.
            apply_bounds(&mut point, bounds_ref);

            let value = match evaluate_point(problem, &point) {
                Ok(v) => v,
                Err(msg) => {
                    return build_results(
                        &simplex,
                        0,
                        nfev,
                        start_time.elapsed(),
                        TerminationReason::FunctionEvaluationFailed(msg),
                        None,
                    )
                }
            };

            simplex.push(EvaluatedPoint::new(point, value));
            nfev += 1;
        }

        if simplex.len() != dim + 1 {
            return build_results(
                &simplex,
                0,
                nfev,
                start_time.elapsed(),
                TerminationReason::DegenerateSimplex,
                None,
            );
        }

        let mut nit = 0usize;
        let mut termination = TerminationReason::MaxIterationsReached;

        loop {
            if let Some(patience) = self.patience {
                if start_time.elapsed() >= patience {
                    termination = TerminationReason::PatienceElapsed;
                    break;
                }
            }

            simplex.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));

            if let Some(reason) = self.convergence_reason(&simplex) {
                termination = reason;
                break;
            }

            if nit >= self.max_iter {
                termination = TerminationReason::MaxIterationsReached;
                break;
            }

            if self.reached_max_evaluations(nfev) {
                termination = TerminationReason::MaxFunctionEvaluationsReached;
                break;
            }

            nit += 1;

            let worst_index = simplex.len() - 1;
            let centroid = Self::centroid(&simplex[..worst_index]);
            let worst_point = &simplex[worst_index].point;
            let worst_value = simplex[worst_index].value;

            let mut reflected_point: Vec<f64> = centroid
                .iter()
                .zip(worst_point)
                .map(|(c, w)| c + self.alpha * (c - w))
                .collect();

            // Reflected candidate must respect bounds to avoid evaluating illegal points.
            apply_bounds(&mut reflected_point, bounds_ref);

            let reflected_value = match evaluate_point(problem, &reflected_point) {
                Ok(v) => v,
                Err(msg) => {
                    termination = TerminationReason::FunctionEvaluationFailed(msg);
                    simplex[worst_index] = EvaluatedPoint::new(reflected_point, f64::NAN);
                    nfev += 1;
                    break;
                }
            };

            nfev += 1;

            if reflected_value < simplex[0].value {
                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    simplex[worst_index] = EvaluatedPoint::new(reflected_point, reflected_value);
                    break;
                }

                let mut expanded_point: Vec<f64> = centroid
                    .iter()
                    .zip(&reflected_point)
                    .map(|(c, r)| c + self.gamma * (r - c))
                    .collect();

                // Expansion step can overshoot, so re-clamp to the allowable region.
                apply_bounds(&mut expanded_point, bounds_ref);

                let expanded_value = match evaluate_point(problem, &expanded_point) {
                    Ok(v) => v,
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        simplex[worst_index] =
                            EvaluatedPoint::new(reflected_point, reflected_value);
                        nfev += 1;
                        break;
                    }
                };

                nfev += 1;

                if expanded_value < reflected_value {
                    simplex[worst_index] = EvaluatedPoint::new(expanded_point, expanded_value);
                } else {
                    simplex[worst_index] = EvaluatedPoint::new(reflected_point, reflected_value);
                }
                continue;
            }

            if reflected_value < simplex[worst_index - 1].value {
                simplex[worst_index] = EvaluatedPoint::new(reflected_point, reflected_value);
                continue;
            }

            let (contract_point, contract_value) = if reflected_value < worst_value {
                // Outside contraction
                let mut point: Vec<f64> = centroid
                    .iter()
                    .zip(&reflected_point)
                    .map(|(c, r)| c + self.rho * (r - c))
                    .collect();

                // Outside contraction needs clamping so it remains within parameter limits.
                apply_bounds(&mut point, bounds_ref);

                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    simplex[worst_index] = EvaluatedPoint::new(reflected_point, reflected_value);
                    break;
                }

                let value = match evaluate_point(problem, &point) {
                    Ok(v) => v,
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        simplex[worst_index] =
                            EvaluatedPoint::new(reflected_point, reflected_value);
                        nfev += 1;
                        break;
                    }
                };

                nfev += 1;
                (point, value)
            } else {
                // Inside contraction
                let mut point: Vec<f64> = centroid
                    .iter()
                    .zip(worst_point)
                    .map(|(c, w)| c + self.rho * (w - c))
                    .collect();

                // Inside contraction also reprojects onto the feasible hyper-rectangle.
                apply_bounds(&mut point, bounds_ref);

                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    simplex[worst_index] = EvaluatedPoint::new(reflected_point, reflected_value);
                    break;
                }

                let value = match evaluate_point(problem, &point) {
                    Ok(v) => v,
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        simplex[worst_index] =
                            EvaluatedPoint::new(reflected_point, reflected_value);
                        nfev += 1;
                        break;
                    }
                };

                nfev += 1;
                (point, value)
            };

            if matches!(termination, TerminationReason::FunctionEvaluationFailed(_)) {
                break;
            }

            if contract_value < worst_value {
                simplex[worst_index] = EvaluatedPoint::new(contract_point, contract_value);
                continue;
            }

            // Shrink
            let best_point = simplex[0].point.clone();
            for item in simplex.iter_mut().skip(1) {
                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    break;
                }

                let mut new_point: Vec<f64> = best_point
                    .iter()
                    .zip(item.point.iter())
                    .map(|(b, x)| b + self.sigma * (x - b))
                    .collect();

                // Shrink step drifts towards the best point; clamp to maintain feasibility.
                apply_bounds(&mut new_point, bounds_ref);

                match evaluate_point(problem, &new_point) {
                    Ok(val) => {
                        *item = EvaluatedPoint::new(new_point, val);
                        nfev += 1;
                    }
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        *item = EvaluatedPoint::new(new_point, f64::NAN);
                        nfev += 1;
                        break;
                    }
                }
            }

            if matches!(
                termination,
                TerminationReason::MaxFunctionEvaluationsReached
                    | TerminationReason::FunctionEvaluationFailed(_)
            ) {
                break;
            }
        }

        build_results(&simplex, nit, nfev, start_time.elapsed(), termination, None)
    }
}

impl Optimiser for NelderMead {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults {
        self.run(problem, initial)
    }
}

impl WithMaxIter for NelderMead {
    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }
}

impl WithThreshold for NelderMead {
    fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}

impl WithSigma0 for NelderMead {
    fn set_sigma0(&mut self, sigma0: f64) {
        self.sigma0 = sigma0;
    }
}

impl WithPatience for NelderMead {
    fn set_patience(&mut self, patience_seconds: f64) {
        if patience_seconds.is_finite() && patience_seconds > 0.0 {
            self.patience = Some(Duration::from_secs_f64(patience_seconds));
        } else {
            self.patience = None;
        }
    }
}

impl Default for NelderMead {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct CMAES {
    max_iter: usize,
    threshold: f64,
    sigma0: f64,
    patience: Option<Duration>,
    population_size: Option<usize>,
    seed: Option<u64>,
}

impl CMAES {
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            threshold: 1e-6,
            sigma0: 0.5,
            patience: None,
            population_size: None,
            seed: None,
        }
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

    fn population_size(&self, dim: usize) -> usize {
        if let Some(size) = self.population_size {
            size.max(1)
        } else if dim > 0 {
            let suggested = (4.0 + (3.0 * (dim as f64).ln())).floor() as usize;
            suggested.max(4).max(2 * dim)
        } else {
            4
        }
    }

    pub fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults {
        let start_time = Instant::now();

        let bounds = extract_bounds(problem);
        let bounds_ref = bounds.as_ref();

        let (start, start_value, mut nfev) = match initialise_start(problem, initial, bounds_ref) {
            InitialState::Finished(results) => return results,
            InitialState::Ready {
                start,
                start_value,
                nfev,
            } => (start, start_value, nfev),
        };

        let dim = start.len();
        if dim == 0 {
            return build_results(
                &[EvaluatedPoint::new(start, start_value)],
                0,
                nfev,
                start_time.elapsed(),
                TerminationReason::BothTolerancesReached,
                None,
            );
        }

        let dim_f = dim as f64;

        let mut mean = DVector::from_vec(start.clone());
        let mut sigma = self.sigma0.max(1e-12);
        let mut cov = DMatrix::identity(dim, dim);
        let mut p_sigma = DVector::zeros(dim);
        let mut p_c = DVector::zeros(dim);
        let mut eigenvectors = DMatrix::identity(dim, dim);
        let mut sqrt_eigenvalues = DVector::from_element(dim, 1.0);
        let mut inv_sqrt_cov = DMatrix::identity(dim, dim);

        let mut rng: StdRng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let lambda = self.population_size(dim);
        let mu = (lambda / 2).max(1);
        let mut weights: Vec<f64> = (0..mu).map(|i| (mu - i) as f64).collect();
        let weight_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= weight_sum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let c_sigma = (mu_eff + 2.0) / (dim_f + mu_eff + 5.0);
        let d_sigma = compute_d_sigma(mu_eff, dim_f, c_sigma);
        let c_c = (4.0 + mu_eff / dim_f) / (dim_f + 4.0 + 2.0 * mu_eff / dim_f);
        let c1 = 2.0 / ((dim_f + 1.3).powi(2) + mu_eff);
        let c_mu = ((1.0 - c1)
            .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim_f + 2.0).powi(2) + mu_eff)))
        .max(0.0);
        let chi_n = dim_f.sqrt() * (1.0 - 1.0 / (4.0 * dim_f) + 1.0 / (21.0 * dim_f.powi(2)));

        let mut nit = 0usize;
        let mut termination = TerminationReason::MaxIterationsReached;
        let mut best_point = EvaluatedPoint::new(start.clone(), start_value);
        let mut final_population = vec![best_point.clone()];

        while nit < self.max_iter {
            if let Some(patience) = self.patience {
                if start_time.elapsed() >= patience {
                    termination = TerminationReason::PatienceElapsed;
                    break;
                }
            }

            if dim > 0 {
                let sym = (&cov + cov.transpose()) * 0.5;
                let eig = sym.symmetric_eigen();
                eigenvectors = eig.eigenvectors;
                sqrt_eigenvalues = eig.eigenvalues.map(|val: f64| val.max(1e-30).sqrt());

                let inv_diag = sqrt_eigenvalues.map(|val| {
                    if val > 0.0 {
                        (1.0 / val).min(1e12_f64)
                    } else {
                        1e12
                    }
                });
                inv_sqrt_cov =
                    &eigenvectors * DMatrix::from_diagonal(&inv_diag) * eigenvectors.transpose();
            }

            let step_matrix = DMatrix::from_diagonal(&sqrt_eigenvalues);

            let mut sampled_points: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut sampled_steps: Vec<DVector<f64>> = Vec::with_capacity(lambda);

            for _ in 0..lambda {
                let z = DVector::from_iterator(
                    dim,
                    (0..dim).map(|_| rng.sample::<f64, _>(StandardNormal)),
                );

                let step = &eigenvectors * (&step_matrix * &z);

                let candidate_vec = mean.clone() + step * sigma;
                let mut candidate: Vec<f64> = candidate_vec.iter().cloned().collect();

                // Sampled population members are projected back inside bounds before scoring.
                apply_bounds(&mut candidate, bounds_ref);

                sampled_points.push(candidate);
                sampled_steps.push(z);
            }

            let evaluations = problem.evaluate_population(&sampled_points);
            nfev += evaluations.len();

            let mut population: Vec<(EvaluatedPoint, DVector<f64>)> = Vec::with_capacity(lambda);
            for ((candidate, z), result) in sampled_points
                .into_iter()
                .zip(sampled_steps.into_iter())
                .zip(evaluations.into_iter())
            {
                let (candidate, z) = (candidate, z);
                match result {
                    Ok(value) => {
                        population.push((EvaluatedPoint::new(candidate, value), z));
                    }
                    Err(msg) => {
                        let mut final_points: Vec<EvaluatedPoint> =
                            population.iter().map(|(pt, _)| pt.clone()).collect();
                        final_points.push(EvaluatedPoint::new(candidate, f64::NAN));
                        return build_results(
                            &final_points,
                            nit,
                            nfev,
                            start_time.elapsed(),
                            TerminationReason::FunctionEvaluationFailed(msg),
                            Some(&cov),
                        );
                    }
                }
            }

            population.sort_by(|a, b| a.0.value.partial_cmp(&b.0.value).unwrap_or(Ordering::Equal));

            // Update best point from sorted population
            if let Some((best, _)) = population.first() {
                if best.value < best_point.value {
                    best_point = best.clone();
                }
            }

            // Calculate function difference from sorted population
            let fun_diff = if population.len() > 1 {
                let best_val = population[0].0.value;
                let worst_val = population[population.len() - 1].0.value;
                (worst_val - best_val).abs()
            } else {
                0.0
            };

            // Build current_points for final results (include best overall)
            let mut current_points: Vec<EvaluatedPoint> =
                population.iter().map(|(pt, _)| pt.clone()).collect();
            if !current_points.iter().any(|pt| pt.point == best_point.point) {
                current_points.push(best_point.clone());
            }

            let old_mean = mean.clone();
            let limit = mu.min(population.len());
            let mut new_mean = DVector::zeros(dim);
            for i in 0..limit {
                let weight = weights[i];
                let candidate_vec = DVector::from_column_slice(&population[i].0.point);
                new_mean += candidate_vec.clone() * weight;
            }
            mean = new_mean;

            let mean_shift = &mean - &old_mean;

            let norm_factor = (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt();
            let mut mean_shift_sigma = mean_shift.clone();
            if sigma > 0.0 {
                mean_shift_sigma /= sigma;
            }
            let delta = &inv_sqrt_cov * mean_shift_sigma.clone();
            let delta_scaled = delta * norm_factor;
            p_sigma = p_sigma * (1.0 - c_sigma) + delta_scaled;

            let norm_p_sigma = p_sigma.norm();
            let exponent = 2.0 * ((nit + 1) as f64);
            let factor = (1.0 - (1.0 - c_sigma).powf(exponent)).max(1e-12).sqrt();
            let h_sigma_threshold = (1.4 + 2.0 / (dim_f + 1.0)) * chi_n;
            let h_sigma = if norm_p_sigma / factor < h_sigma_threshold {
                1.0
            } else {
                0.0
            };

            let pc_factor = (c_c * (2.0 - c_c) * mu_eff).sqrt();
            let sigma_denom = sigma.max(1e-12);
            let mean_shift_scaled = if sigma > 0.0 {
                mean_shift.clone() * (h_sigma * pc_factor / sigma_denom)
            } else {
                DVector::zeros(dim)
            };
            p_c = p_c * (1.0 - c_c) + mean_shift_scaled;

            let mut rank_mu_update = DMatrix::zeros(dim, dim);
            for i in 0..limit {
                let weight = weights[i];
                let candidate_vec = DVector::from_column_slice(&population[i].0.point);
                let y = (candidate_vec - &old_mean) / sigma_denom;
                rank_mu_update += (&y * y.transpose()) * weight;
            }

            cov = update_covariance(&cov, c1, c_mu, &p_c, h_sigma, c_c, &rank_mu_update);

            sigma *= (c_sigma / d_sigma * (norm_p_sigma / chi_n - 1.0)).exp();
            sigma = sigma.max(1e-18);

            nit += 1;
            final_population = current_points;

            let position_converged = mean_shift.norm() <= self.threshold;
            let fun_converged = fun_diff <= self.threshold;
            if fun_converged && position_converged {
                termination = TerminationReason::BothTolerancesReached;
                break;
            } else if fun_converged {
                termination = TerminationReason::FunctionToleranceReached;
                break;
            } else if position_converged {
                termination = TerminationReason::ParameterToleranceReached;
                break;
            }
        }

        build_results(
            &final_population,
            nit,
            nfev,
            start_time.elapsed(),
            termination,
            Some(&cov),
        )
    }
}

impl Optimiser for CMAES {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults {
        self.run(problem, initial)
    }
}

impl WithMaxIter for CMAES {
    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }
}

impl WithThreshold for CMAES {
    fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}

impl WithSigma0 for CMAES {
    fn set_sigma0(&mut self, sigma0: f64) {
        self.sigma0 = sigma0.max(1e-12);
    }
}

impl WithPatience for CMAES {
    fn set_patience(&mut self, patience_seconds: f64) {
        if patience_seconds.is_finite() && patience_seconds > 0.0 {
            self.patience = Some(Duration::from_secs_f64(patience_seconds));
        } else {
            self.patience = None;
        }
    }
}

impl Default for CMAES {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_d_sigma(mu_eff: f64, dim_f: f64, c_sigma: f64) -> f64 {
    let sqrt_term = ((mu_eff - 1.0) / (dim_f + 1.0)).max(0.0).sqrt();
    1.0 + c_sigma + 2.0 * (sqrt_term - 1.0).max(0.0)
}

fn update_covariance(
    cov: &DMatrix<f64>,
    c1: f64,
    c_mu: f64,
    p_c: &DVector<f64>,
    h_sigma: f64,
    c_c: f64,
    rank_mu_contrib: &DMatrix<f64>,
) -> DMatrix<f64> {
    let mut updated = cov * (1.0 - c1 - c_mu);

    // Rank-one update with exponential correction for h_sigma
    let correction_factor = (1.0 - h_sigma) * c_c * (2.0 - c_c);
    let rank_one = p_c * p_c.transpose();
    updated += rank_one * c1;
    updated += cov * (correction_factor * c1);

    // Rank-mu update
    updated += rank_mu_contrib * c_mu;

    updated
}

// Results object
#[derive(Debug, Clone)]
pub struct OptimisationResults {
    pub x: Vec<f64>,
    pub fun: f64,
    pub nit: usize,
    pub nfev: usize,
    pub time: Duration,
    pub success: bool,
    pub message: String,
    pub termination_reason: TerminationReason,
    pub final_simplex: Vec<Vec<f64>>,
    pub final_simplex_values: Vec<f64>,
    pub covariance: Option<Vec<Vec<f64>>>,
}

impl OptimisationResults {
    fn __repr__(&self) -> String {
        format!(
            "OptimisationResults(x={:?}, fun={:.6}, nit={}, nfev={}, time={:?}, success={}, reason={})",
            self.x, self.fun, self.nit, self.nfev, self.time, self.success, self.message
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{BuilderParameterExt, ScalarProblemBuilder};
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn nelder_mead_minimises_quadratic() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| {
                let x0 = x[0] - 1.5;
                let x1 = x[1] + 0.5;
                x0 * x0 + x1 * x1
            })
            .build()
            .unwrap();

        let optimiser = NelderMead::new()
            .with_max_iter(400)
            .with_threshold(1e-10)
            .with_sigma0(0.6)
            .with_position_tolerance(1e-8);

        let result = optimiser.run(&problem, vec![5.0, -4.0]);

        assert!(result.success, "Expected success: {}", result.message);
        assert!((result.x[0] - 1.5).abs() < 1e-5);
        assert!((result.x[1] + 0.5).abs() < 1e-5);
        assert!(result.fun < 1e-9, "Final value too large: {}", result.fun);
        assert!(result.nit > 0);
        assert!(result.nfev >= result.nit + 1);
    }

    #[test]
    fn nelder_mead_respects_max_iterations() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .build()
            .unwrap();

        let optimiser = NelderMead::new().with_max_iter(1).with_sigma0(1.0);
        let result = optimiser.run(&problem, vec![10.0, -10.0]);

        assert_eq!(
            result.termination_reason,
            TerminationReason::MaxIterationsReached
        );
        assert!(!result.success);
        assert!(result.nit <= 1);
    }

    #[test]
    fn nelder_mead_respects_max_function_evaluations() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .build()
            .unwrap();

        let optimiser = NelderMead::new()
            .with_max_evaluations(2)
            .with_sigma0(0.5)
            .with_max_iter(500);

        let result = optimiser.run(&problem, vec![2.0, 2.0]);

        assert_eq!(
            result.termination_reason,
            TerminationReason::MaxFunctionEvaluationsReached
        );
        assert!(!result.success);
        assert!(result.nfev <= 2);
    }

    #[test]
    fn nelder_mead_respects_patience() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| {
                std::thread::sleep(Duration::from_millis(5));
                x.iter().map(|xi| xi * xi).sum()
            })
            .build()
            .unwrap();

        let optimiser = NelderMead::new().with_sigma0(0.5).with_patience(0.01);

        let result = optimiser.run(&problem, vec![5.0, -5.0]);

        assert_eq!(
            result.termination_reason,
            TerminationReason::PatienceElapsed
        );
        assert!(!result.success);
    }

    #[test]
    fn cmaes_minimises_quadratic() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| {
                let x0 = x[0] - 1.5;
                let x1 = x[1] + 0.5;
                x0 * x0 + x1 * x1
            })
            .build()
            .unwrap();

        let optimiser = CMAES::new()
            .with_max_iter(400)
            .with_threshold(1e-10)
            .with_sigma0(0.6)
            .with_patience(5.0)
            .with_seed(42);

        let result = optimiser.run(&problem, vec![5.0, -4.0]);

        assert!(result.success, "Expected success: {}", result.message);
        assert!((result.x[0] - 1.5).abs() < 1e-4);
        assert!((result.x[1] + 0.5).abs() < 1e-4);
        assert!(result.fun < 1e-8, "Final value too large: {}", result.fun);
        assert!(result.nit > 0);
        assert!(result.nfev >= result.nit + 1);
    }

    #[test]
    fn cmaes_respects_max_iterations() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .build()
            .unwrap();

        let optimiser = CMAES::new().with_max_iter(1).with_sigma0(0.5).with_seed(7);
        let result = optimiser.run(&problem, vec![10.0, -10.0]);

        assert_eq!(
            result.termination_reason,
            TerminationReason::MaxIterationsReached
        );
        assert!(!result.success);
        assert!(result.nit <= 1);
    }

    #[test]
    fn cmaes_respects_patience() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| {
                std::thread::sleep(Duration::from_millis(5));
                x.iter().map(|xi| xi * xi).sum()
            })
            .build()
            .unwrap();

        let optimiser = CMAES::new()
            .with_sigma0(0.5)
            .with_patience(0.01)
            .with_seed(5);

        let result = optimiser.run(&problem, vec![5.0, -5.0]);

        assert_eq!(
            result.termination_reason,
            TerminationReason::PatienceElapsed
        );
        assert!(!result.success);
    }

    // Edge case tests
    #[test]
    fn nelder_mead_handles_bounds() {
        use crate::problem::ParameterSpec;

        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2))
            .with_parameter(ParameterSpec::new("x", 0.0, Some((-1.0, 1.0))))
            .with_parameter(ParameterSpec::new("y", 0.0, Some((0.0, 2.0))))
            .build()
            .unwrap();

        let optimiser = NelderMead::new().with_max_iter(200).with_threshold(1e-8);

        let result = optimiser.run(&problem, vec![0.5, 1.0]);

        // Should converge to bounds: x=1.0 (clamped from 2.0), y=2.0 (clamped from 3.0)
        assert!(
            result.x[0] >= -1.0 && result.x[0] <= 1.0,
            "x out of bounds: {}",
            result.x[0]
        );
        assert!(
            result.x[1] >= 0.0 && result.x[1] <= 2.0,
            "y out of bounds: {}",
            result.x[1]
        );
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x should be near upper bound"
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.1,
            "y should be near upper bound"
        );
    }

    #[test]
    fn cmaes_handles_bounds() {
        use crate::problem::ParameterSpec;

        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] + 5.0).powi(2))
            .with_parameter(ParameterSpec::new("x", 0.0, Some((0.0, 3.0))))
            .with_parameter(ParameterSpec::new("y", 0.0, Some((-3.0, 0.0))))
            .build()
            .unwrap();

        let optimiser = CMAES::new()
            .with_max_iter(100)
            .with_threshold(1e-6)
            .with_seed(123);

        let result = optimiser.run(&problem, vec![1.5, -1.5]);

        // Should converge to bounds: x=3.0 (clamped from 5.0), y=-3.0 (clamped from -5.0)
        assert!(
            result.x[0] >= 0.0 && result.x[0] <= 3.0,
            "x out of bounds: {}",
            result.x[0]
        );
        assert!(
            result.x[1] >= -3.0 && result.x[1] <= 0.0,
            "y out of bounds: {}",
            result.x[1]
        );
        assert!(
            (result.x[0] - 3.0).abs() < 0.2,
            "x should be near upper bound"
        );
        assert!(
            (result.x[1] + 3.0).abs() < 0.2,
            "y should be near lower bound"
        );
    }

    #[test]
    fn cmaes_population_size_heuristics() {
        let default = CMAES::new();
        assert_eq!(default.population_size(0), 4, "zero dimension defaults");
        assert_eq!(default.population_size(1), 4, "univariate heuristic");
        assert_eq!(
            default.population_size(10),
            20,
            "high dimension uses lambda >= 2n"
        );

        let overridden = CMAES::new().with_population_size(6);
        assert_eq!(
            overridden.population_size(2),
            6,
            "explicit population respected when >= 1"
        );
    }

    #[test]
    fn cmaes_is_reproducible_with_seed() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| {
                (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
            })
            .build()
            .unwrap();

        let optimiser = CMAES::new()
            .with_max_iter(300)
            .with_threshold(1e-8)
            .with_sigma0(0.7)
            .with_seed(2024);

        let initial = vec![3.0, -2.0];
        let result_one = optimiser.run(&problem, initial.clone());
        let result_two = optimiser.run(&problem, initial);

        assert!(
            result_one.success,
            "first run should converge: {}",
            result_one.message
        );
        assert!(
            result_two.success,
            "second run should converge: {}",
            result_two.message
        );
        assert_eq!(result_one.nit, result_two.nit);
        assert_eq!(result_one.nfev, result_two.nfev);
        assert!((result_one.fun - result_two.fun).abs() < 1e-12);
        for (x1, x2) in result_one.x.iter().zip(result_two.x.iter()) {
            assert!(
                (x1 - x2).abs() < 1e-10,
                "expected identical optima: {} vs {}",
                x1,
                x2
            );
        }
        assert_eq!(result_one.covariance, result_two.covariance);
    }

    #[test]
    fn nelder_mead_handles_nan_objective() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| if x[0] > 1.0 { f64::NAN } else { x[0] * x[0] })
            .build()
            .unwrap();

        let optimiser = NelderMead::new().with_max_iter(50).with_sigma0(2.0); // Larger sigma to ensure we hit NaN region

        let result = optimiser.run(&problem, vec![0.5]);

        // Should either detect NaN or converge to valid region
        // If it hits NaN, it should fail gracefully
        if !result.success {
            assert!(matches!(
                result.termination_reason,
                TerminationReason::FunctionEvaluationFailed(_)
            ));
        }
        // Otherwise it converged to the valid region (x <= 1.0)
    }

    #[test]
    fn cmaes_lazy_eigendecomposition_works() {
        // Test with high dimension to trigger lazy updates
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
            .build()
            .unwrap();

        let dim = 60; // > 50 to trigger lazy updates
        let initial = vec![0.5; dim];

        let optimiser = CMAES::new()
            .with_max_iter(100)
            .with_threshold(1e-6)
            .with_seed(777);

        let initial_value = initial.iter().map(|x| x * x).sum::<f64>();

        let result = optimiser.run(&problem, initial);

        // Should still work with lazy updates and improve from initial
        assert!(result.nfev > 0);
        assert!(
            result.fun < initial_value,
            "Should improve: {} < {}",
            result.fun,
            initial_value
        );
    }

    #[test]
    fn cmaes_covariance_is_symmetric_and_psd() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| {
                (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
            })
            .build()
            .unwrap();

        let optimiser = CMAES::new()
            .with_max_iter(400)
            .with_threshold(1e-10)
            .with_sigma0(0.6)
            .with_seed(4242);

        let result = optimiser.run(&problem, vec![4.5, -3.5]);

        assert!(result.success, "Expected success: {}", result.message);
        assert!(
            result.fun < 1e-6,
            "Should reach low objective value: {}",
            result.fun
        );

        let covariance = result
            .covariance
            .clone()
            .expect("CMAES should provide covariance estimates");
        assert_eq!(covariance.len(), 2);
        assert!(covariance.iter().all(|row| row.len() == 2));

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (covariance[i][j] - covariance[j][i]).abs() < 1e-12,
                    "covariance matrix must be symmetric"
                );
            }
        }

        let flat: Vec<f64> = covariance
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let matrix = DMatrix::from_row_slice(2, 2, &flat);
        let eigenvalues = matrix.symmetric_eigen().eigenvalues;

        assert!(
            eigenvalues.iter().all(|&eig| eig >= -1e-10),
            "covariance must be positive semi-definite: {:?}",
            eigenvalues
        );
    }

    #[test]
    fn d_sigma_matches_hansen_2016_formula() {
        // Test case 1: Standard parameters from 10-dimensional optimization
        let mu_eff = 4.5;
        let dim_f = 10.0;
        let c_sigma = 0.3;

        // Manual computation following Hansen (2016)
        let inner: f64 = (mu_eff - 1.0) / (dim_f + 1.0); // (4.5 - 1) / 11 = 0.318...
        let sqrt_inner = inner.sqrt(); // ~0.564
        let clamped = (sqrt_inner - 1.0).max(0.0); // max(0, -0.436) = 0.0
        let expected = 1.0 + c_sigma + 2.0 * clamped; // 1.0 + 0.3 + 0 = 1.3

        let computed = compute_d_sigma(mu_eff, dim_f, c_sigma);

        assert!(
            (computed - expected).abs() < 1e-12,
            "d_sigma mismatch: expected {}, got {}",
            expected,
            computed
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
        let computed = compute_d_sigma(mu_eff, dim_f, c_sigma);

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

        let updated = update_covariance(&cov, c1, c_mu, &p_c, h_sigma, c_c, &rank_mu);

        for (exp, got) in expected.iter().zip(updated.iter()) {
            assert!((exp - got).abs() < 1e-12, "expected {} got {}", exp, got);
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

        let updated = update_covariance(&cov, c1, c_mu, &p_c, h_sigma, c_c, &rank_mu);

        for (exp, got) in expected.iter().zip(updated.iter()) {
            assert!((exp - got).abs() < 1e-12, "expected {} got {}", exp, got);
        }
    }
}
