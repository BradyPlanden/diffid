use crate::problem::Problem;
use std::cmp::Ordering;
use std::fmt;
use std::time::{Duration, Instant};

// Core behaviour shared by all optimisers
pub trait Optimiser {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults;
}

// Optimiser traits
pub trait WithMaxIter: Sized {
    fn set_max_iter(&mut self, max_iter: usize);
    fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.set_max_iter(max_iter);
        self
    }
}

pub trait WithThreshold: Sized {
    fn set_threshold(&mut self, threshold: f64);
    fn with_threshold(mut self, threshold: f64) -> Self {
        self.set_threshold(threshold);
        self
    }
}

pub trait WithSigma0: Sized {
    fn set_sigma0(&mut self, sigma0: f64);
    fn with_sigma0(mut self, sigma0: f64) -> Self {
        self.set_sigma0(sigma0);
        self
    }
}

pub trait WithPatience: Sized {
    fn set_patience(&mut self, patience_seconds: f64);
    fn with_patience(mut self, patience_seconds: f64) -> Self {
        self.set_patience(patience_seconds);
        self
    }
}

#[derive(Debug, Clone)]
struct SimplexVertex {
    point: Vec<f64>,
    value: f64,
}

impl SimplexVertex {
    fn new(point: Vec<f64>, value: f64) -> Self {
        Self { point, value }
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

    fn convergence_reason(&self, simplex: &[SimplexVertex]) -> Option<TerminationReason> {
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

    fn build_results(
        simplex: &[SimplexVertex],
        nit: usize,
        nfev: usize,
        reason: TerminationReason,
    ) -> OptimisationResults {
        let mut ordered = simplex.to_vec();
        ordered.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));

        let best = ordered
            .first()
            .cloned()
            .unwrap_or_else(|| SimplexVertex::new(Vec::new(), f64::NAN));

        let final_simplex = ordered.iter().map(|v| v.point.clone()).collect();
        let final_simplex_values = ordered.iter().map(|v| v.value).collect();

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
            success,
            message,
            termination_reason: reason,
            final_simplex,
            final_simplex_values,
        }
    }

    fn evaluate(problem: &Problem, point: &[f64]) -> Result<f64, String> {
        problem.evaluate(point)
    }

    fn centroid(simplex: &[SimplexVertex]) -> Vec<f64> {
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

        let start = if !initial.is_empty() {
            initial
        } else {
            let dim = problem.dimension();
            vec![0.0; dim]
        };

        let dim = start.len();

        if dim == 0 {
            let value = match Self::evaluate(problem, &start) {
                Ok(v) => v,
                Err(msg) => {
                    return Self::build_results(
                        &[SimplexVertex::new(Vec::new(), f64::NAN)],
                        0,
                        0,
                        TerminationReason::FunctionEvaluationFailed(msg),
                    )
                }
            };

            return Self::build_results(
                &[SimplexVertex::new(start, value)],
                0,
                1,
                TerminationReason::BothTolerancesReached,
            );
        }

        let start_value = match Self::evaluate(problem, &start) {
            Ok(v) => v,
            Err(msg) => {
                return Self::build_results(
                    &[SimplexVertex::new(start, f64::NAN)],
                    0,
                    1,
                    TerminationReason::FunctionEvaluationFailed(msg),
                )
            }
        };

        let mut simplex = vec![SimplexVertex::new(start.clone(), start_value)];
        let mut nfev = 1usize;

        for i in 0..dim {
            if self.reached_max_evaluations(nfev) {
                return Self::build_results(
                    &simplex,
                    0,
                    nfev,
                    TerminationReason::MaxFunctionEvaluationsReached,
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

            let value = match Self::evaluate(problem, &point) {
                Ok(v) => v,
                Err(msg) => {
                    return Self::build_results(
                        &simplex,
                        0,
                        nfev,
                        TerminationReason::FunctionEvaluationFailed(msg),
                    )
                }
            };

            simplex.push(SimplexVertex::new(point, value));
            nfev += 1;
        }

        if simplex.len() != dim + 1 {
            return Self::build_results(&simplex, 0, nfev, TerminationReason::DegenerateSimplex);
        }

        let mut nit = 0usize;
        if let Some(patience) = self.patience {
            if start_time.elapsed() >= patience {
                return Self::build_results(
                    &simplex,
                    nit,
                    nfev,
                    TerminationReason::PatienceElapsed,
                );
            }
        }
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
            let worst = simplex[worst_index].clone();

            let reflected_point: Vec<f64> = centroid
                .iter()
                .zip(&worst.point)
                .map(|(c, w)| c + self.alpha * (c - w))
                .collect();

            let reflected_value = match Self::evaluate(problem, &reflected_point) {
                Ok(v) => v,
                Err(msg) => {
                    termination = TerminationReason::FunctionEvaluationFailed(msg);
                    simplex[worst_index] = SimplexVertex::new(reflected_point, f64::NAN);
                    nfev += 1;
                    break;
                }
            };

            nfev += 1;

            if reflected_value < simplex[0].value {
                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                    break;
                }

                let expanded_point: Vec<f64> = centroid
                    .iter()
                    .zip(&reflected_point)
                    .map(|(c, r)| c + self.gamma * (r - c))
                    .collect();

                let expanded_value = match Self::evaluate(problem, &expanded_point) {
                    Ok(v) => v,
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                        nfev += 1;
                        break;
                    }
                };

                nfev += 1;

                if expanded_value < reflected_value {
                    simplex[worst_index] = SimplexVertex::new(expanded_point, expanded_value);
                } else {
                    simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                }
                continue;
            }

            if reflected_value < simplex[worst_index - 1].value {
                simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                continue;
            }

            let (contract_point, contract_value) = if reflected_value < worst.value {
                // Outside contraction
                let point: Vec<f64> = centroid
                    .iter()
                    .zip(&reflected_point)
                    .map(|(c, r)| c + self.rho * (r - c))
                    .collect();

                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                    break;
                }

                let value = match Self::evaluate(problem, &point) {
                    Ok(v) => v,
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                        nfev += 1;
                        break;
                    }
                };

                nfev += 1;
                (point, value)
            } else {
                // Inside contraction
                let point: Vec<f64> = centroid
                    .iter()
                    .zip(&worst.point)
                    .map(|(c, w)| c + self.rho * (w - c))
                    .collect();

                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                    break;
                }

                let value = match Self::evaluate(problem, &point) {
                    Ok(v) => v,
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        simplex[worst_index] = SimplexVertex::new(reflected_point, reflected_value);
                        nfev += 1;
                        break;
                    }
                };

                nfev += 1;
                (point, value)
            };

            if termination != TerminationReason::MaxIterationsReached
                && matches!(termination, TerminationReason::FunctionEvaluationFailed(_))
            {
                break;
            }

            if contract_value < worst.value {
                simplex[worst_index] = SimplexVertex::new(contract_point, contract_value);
                continue;
            }

            // Shrink
            let best_point = simplex[0].point.clone();
            for idx in 1..simplex.len() {
                if self.reached_max_evaluations(nfev) {
                    termination = TerminationReason::MaxFunctionEvaluationsReached;
                    break;
                }

                let new_point: Vec<f64> = best_point
                    .iter()
                    .zip(simplex[idx].point.iter())
                    .map(|(b, x)| b + self.sigma * (x - b))
                    .collect();

                match Self::evaluate(problem, &new_point) {
                    Ok(val) => {
                        simplex[idx] = SimplexVertex::new(new_point, val);
                        nfev += 1;
                    }
                    Err(msg) => {
                        termination = TerminationReason::FunctionEvaluationFailed(msg);
                        simplex[idx] = SimplexVertex::new(new_point, f64::NAN);
                        nfev += 1;
                        break;
                    }
                }
            }

            if !matches!(
                termination,
                TerminationReason::MaxIterationsReached
                    | TerminationReason::FunctionToleranceReached
                    | TerminationReason::ParameterToleranceReached
                    | TerminationReason::BothTolerancesReached
            ) {
                if matches!(
                    termination,
                    TerminationReason::MaxFunctionEvaluationsReached
                        | TerminationReason::FunctionEvaluationFailed(_)
                ) {
                    break;
                }
            }
        }

        Self::build_results(&simplex, nit, nfev, termination)
    }
}

impl Optimiser for NelderMead {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults {
        NelderMead::run(self, problem, initial)
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

// Results object
#[derive(Debug, Clone)]
pub struct OptimisationResults {
    pub x: Vec<f64>,
    pub fun: f64,
    pub nit: usize,
    pub nfev: usize,
    pub success: bool,
    pub message: String,
    pub termination_reason: TerminationReason,
    pub final_simplex: Vec<Vec<f64>>,
    pub final_simplex_values: Vec<f64>,
}

impl OptimisationResults {
    fn __repr__(&self) -> String {
        format!(
            "OptimisationResults(x={:?}, fun={:.6}, nit={}, nfev={}, success={}, reason={})",
            self.x, self.fun, self.nit, self.nfev, self.success, self.message
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::Builder;

    #[test]
    fn nelder_mead_minimises_quadratic() {
        let problem = Builder::new()
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
        let problem = Builder::new()
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
        let problem = Builder::new()
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
        let problem = Builder::new()
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
}
