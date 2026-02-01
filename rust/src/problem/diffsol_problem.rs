use crate::builders::{DiffsolBackend, DiffsolConfig};
use crate::cost::CostMetric;
use crate::problem::{Objective, ProblemError};
use diffsol::error::DiffsolError;
use diffsol::ode_solver::sensitivities::SensitivitiesOdeSolverMethod;
use diffsol::op::Op;
use diffsol::{
    DiffSl, FaerSparseLU, FaerSparseMat, FaerVec, Matrix, MatrixCommon, NalgebraLU, NalgebraMat,
    NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem, Vector,
};
use nalgebra::DMatrix;

use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Index;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "cranelift-backend")]
type CG = diffsol::CraneliftJitModule;
#[cfg(not(feature = "cranelift-backend"))]
type CG = diffsol::LlvmModule;

type DenseEqn = DiffSl<NalgebraMat<f64>, CG>;
type SparseEqn = DiffSl<FaerSparseMat<f64>, CG>;

pub enum DiffsolSimulator {
    Dense(Box<OdeSolverProblem<DenseEqn>>),
    Sparse(Box<OdeSolverProblem<SparseEqn>>),
}

type DenseVector = NalgebraVec<f64>;
type SparseVector = FaerVec<f64>;
type DenseSolver = NalgebraLU<f64>;
type SparseSolver = FaerSparseLU<f64>;

const FAILED_SOLVE_PENALTY: f64 = 1e5;

// Global ID counter for unique objective identification
static NEXT_OBJECTIVE_ID: AtomicUsize = AtomicUsize::new(1);

// Cache structure to hold simulator and its last parameters
struct CachedSimulator {
    simulator: DiffsolSimulator,
    last_params: Vec<f64>,
}

// Thread-local cache: each thread maintains its own cache of simulators
thread_local! {
    static SIMULATOR_CACHE: RefCell<HashMap<usize, CachedSimulator>>
        = RefCell::new(HashMap::new());
}

/// Diffsol Objective
pub struct DiffsolObjective {
    id: usize,
    dsl: String,
    t_span: Vec<f64>,
    data: DMatrix<f64>,
    config: DiffsolConfig,
    costs: Vec<Arc<dyn CostMetric>>,
}

impl DiffsolObjective {
    pub fn new(
        dsl: String,
        t_span: Vec<f64>,
        data: DMatrix<f64>,
        config: DiffsolConfig,
        costs: Vec<Arc<dyn CostMetric>>,
    ) -> Self {
        let id = NEXT_OBJECTIVE_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            dsl,
            t_span,
            data,
            config,
            costs,
        }
    }
    fn build_simulator(&self) -> Result<DiffsolSimulator, ProblemError> {
        match self.config.backend {
            DiffsolBackend::Dense => {
                let diff_system = OdeBuilder::<NalgebraMat<f64>>::new()
                    .atol([self.config.atol])
                    .rtol(self.config.rtol)
                    .build_from_diffsl(&self.dsl)
                    .map_err(|e| {
                        ProblemError::BuildFailed(format!("Failed to build ODE model: {e}"))
                    })?;
                Ok(DiffsolSimulator::Dense(Box::new(diff_system)))
            }
            DiffsolBackend::Sparse => {
                let diff_system = OdeBuilder::<FaerSparseMat<f64>>::new()
                    .atol([self.config.atol])
                    .rtol(self.config.rtol)
                    .build_from_diffsl(&self.dsl)
                    .map_err(|e| {
                        ProblemError::BuildFailed(format!("Failed to build ODE model: {e}"))
                    })?;
                Ok(DiffsolSimulator::Sparse(Box::new(diff_system)))
            }
        }
    }

    fn with_simulator_cached<F, R>(&self, params: &[f64], f: F) -> Result<R, ProblemError>
    where
        F: FnOnce(&mut DiffsolSimulator) -> Result<R, ProblemError>,
    {
        SIMULATOR_CACHE.with(|cache| {
            let mut cache_map = cache.borrow_mut();

            let cached = match cache_map.entry(self.id) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let simulator = self.build_simulator()?;
                    entry.insert(CachedSimulator {
                        simulator,
                        last_params: Vec::new(),
                    })
                }
            };

            if cached.last_params.as_slice() != params {
                match &mut cached.simulator {
                    DiffsolSimulator::Dense(p) => {
                        let ctx = *p.eqn().context();
                        p.eqn_mut()
                            .set_params(&DenseVector::from_vec(params.to_vec(), ctx));
                    }
                    DiffsolSimulator::Sparse(p) => {
                        let ctx = *p.eqn().context();
                        p.eqn_mut()
                            .set_params(&SparseVector::from_vec(params.to_vec(), ctx));
                    }
                }
                cached.last_params = params.to_vec();
            }

            f(&mut cached.simulator)
        })
    }

    /// Helper to solve with panic recovery
    #[inline]
    fn solve_safely<F, T>(solve_fn: F) -> Result<T, ProblemError>
    where
        F: FnOnce() -> Result<T, DiffsolError>,
    {
        catch_unwind(AssertUnwindSafe(solve_fn))
            .map_err(|_| ProblemError::SolverError("Solver panicked".to_string()))?
            .map_err(|e| ProblemError::SolverError(e.to_string()))
    }

    fn build_residuals<M>(&self, solution: &M) -> Result<Vec<f64>, ProblemError>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let sol_rows = solution.nrows(); // n_states
        let sol_cols = solution.ncols(); // n_timepoints
        let sol_size = sol_rows * sol_cols;

        if sol_size != self.data.len() {
            return Err(ProblemError::DimensionMismatch {
                expected: self.data.len(),
                got: sol_size,
            });
        }

        // Solution is organized as (n_states, n_timepoints)
        // Data is organized as (n_timepoints, n_states)
        // We need to iterate with swapped indices
        let mut residuals = Vec::with_capacity(sol_size);
        for time_idx in 0..self.data.nrows() {
            for state_idx in 0..self.data.ncols() {
                residuals.push(solution[(state_idx, time_idx)] - self.data[(time_idx, state_idx)]);
            }
        }

        Ok(residuals)
    }

    #[inline]
    fn calculate_cost<M>(&self, solution: &M) -> Result<f64, ProblemError>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let residuals = self.build_residuals(solution)?;
        let total_cost = self.costs.iter().map(|c| c.evaluate(&residuals)).sum();
        Ok(total_cost)
    }

    fn calculate_cost_with_grad(
        &self,
        solution: &NalgebraMat<f64>,
        sensitivities: &[NalgebraMat<f64>],
    ) -> Result<(f64, Option<Vec<f64>>), ProblemError> {
        let residuals = self.build_residuals(solution)?;

        let mut total_cost = 0.0;
        let mut total_grad: Option<Vec<f64>> = None;

        for metric in &self.costs {
            let (cost, grad) = metric
                .evaluate_with_sensitivities(&residuals, sensitivities)
                .ok_or_else(|| {
                    ProblemError::EvaluationFailed(format!(
                        "Failed to evaluate metric: {}",
                        metric.name()
                    ))
                })?;
            total_cost += cost;

            match &mut total_grad {
                Some(acc) => acc.iter_mut().zip(&grad).for_each(|(a, b)| *a += b),
                None => total_grad = Some(grad),
            }
        }
        Ok((total_cost, total_grad))
    }
}

impl Objective for DiffsolObjective {
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        self.with_simulator_cached(x, |problem| match problem {
            DiffsolSimulator::Dense(p) => {
                let solver_result = p.bdf::<DenseSolver>();
                let Ok(mut solver) = solver_result else {
                    return Ok(FAILED_SOLVE_PENALTY);
                };

                Self::solve_safely(|| solver.solve_dense(&self.t_span)).map_or_else(
                    |_| Ok(FAILED_SOLVE_PENALTY),
                    |solution| self.calculate_cost(&solution),
                )
            }
            DiffsolSimulator::Sparse(p) => {
                let solver_result = p.bdf::<SparseSolver>();
                let Ok(mut solver) = solver_result else {
                    return Ok(FAILED_SOLVE_PENALTY);
                };

                Self::solve_safely(|| solver.solve_dense(&self.t_span)).map_or_else(
                    |_| Ok(FAILED_SOLVE_PENALTY),
                    |solution| self.calculate_cost(&solution),
                )
            }
        })
    }

    fn evaluate_with_gradient(&self, x: &[f64]) -> Result<(f64, Option<Vec<f64>>), ProblemError> {
        self.with_simulator_cached(x, |problem| match problem {
            DiffsolSimulator::Dense(p) => {
                let mut solver = p
                    .bdf_sens::<DenseSolver>()
                    .map_err(|e| ProblemError::SolverError(e.to_string()))?;
                let (solution, sensitivities) =
                    Self::solve_safely(|| solver.solve_dense_sensitivities(&self.t_span))?;

                self.calculate_cost_with_grad(&solution, &sensitivities)
            }
            DiffsolSimulator::Sparse(_p) => Err(ProblemError::EvaluationFailed(
                "Sparse diffsol backend does not currently support gradient evaluation".to_string(),
            )),
        })
    }

    /// Population based evaluation supporting
    /// both sequential and parallel configurations.
    fn evaluate_population(&self, params: &[Vec<f64>]) -> Vec<Result<f64, ProblemError>> {
        if self.config.parallel {
            params.par_iter().map(|x| self.evaluate(x)).collect()
        } else {
            params.iter().map(|x| self.evaluate(x)).collect()
        }
    }

    /// Support parallel evaluation if config.parallel
    /// is set by user.
    fn supports_parallel_evaluation(&self) -> bool {
        self.config.parallel
    }

    fn has_gradient(&self) -> bool {
        // Check the backend config
        matches!(self.config.backend, DiffsolBackend::Dense)
    }
}

impl Drop for DiffsolObjective {
    fn drop(&mut self) {
        SIMULATOR_CACHE.with(|cache| {
            cache.borrow_mut().remove(&self.id);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{GaussianNll, SumSquaredError};

    fn build_logistic_problem(backend: DiffsolBackend) -> DiffsolObjective {
        let dsl = r"
in_i {r = 1, k = 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
";

        let t_span: Vec<f64> = (0..6).map(|i| i as f64 * 0.2).collect();
        let data_values: Vec<f64> = t_span.iter().map(|t| 0.1 * (*t).exp()).collect();
        let data = DMatrix::from_vec(t_span.len(), 1, data_values);
        let config = DiffsolConfig::default().with_backend(backend);

        DiffsolObjective::new(
            dsl.to_string(),
            t_span,
            data,
            config,
            vec![Arc::new(SumSquaredError::default())],
        )
    }

    fn finite_difference<F>(x: &mut [f64], idx: usize, eps: f64, f: F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let original = x[idx];

        x[idx] = original + eps;
        let f_plus = f(x);

        x[idx] = original - eps;
        let f_minus = f(x);

        x[idx] = original;

        (f_plus - f_minus) / (2.0 * eps)
    }

    #[test]
    fn test_gradient_with_empty_sensitivities() {
        let metric = SumSquaredError::default();
        let residuals = vec![1.0, 2.0];
        let sensitivities: Vec<NalgebraMat<f64>> = Vec::new();
        let (cost, grad) = metric
            .evaluate_with_sensitivities(&residuals, &sensitivities)
            .expect("SumSquaredError should support gradient evaluation");
        assert_eq!(cost, 5.0);
        assert!(grad.is_empty());
    }

    #[test]
    fn test_gradient_dimensions_mismatch() {
        let metric = SumSquaredError::default();
        let residuals = vec![1.0, 2.0, 3.0];
        // Build a 2x1 sensitivity matrix (2 elements) which mismatches the 3 residuals
        let triplets = vec![(0, 0, 0.5), (1, 0, 0.5)];
        let wrong_size_sens: NalgebraMat<f64> =
            Matrix::try_from_triplets(2, 1, triplets, diffsol::NalgebraContext).unwrap();

        let result = std::panic::catch_unwind(|| {
            metric
                .evaluate_with_sensitivities(&residuals, &[wrong_size_sens])
                .unwrap();
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_gaussian_nll_gradient_correctness() {
        let variance = 2.0;
        let metric = GaussianNll::new(None, variance);
        let residuals = vec![1.0, -2.0, 0.5];

        // Gradient should be residual/variance
        let sensitivities: Vec<NalgebraMat<f64>> = (0..3)
            .map(|param_idx| {
                let triplets = vec![(param_idx, 0, 1.0)];
                Matrix::try_from_triplets(3, 1, triplets, diffsol::NalgebraContext).unwrap()
            })
            .collect();

        let (_, grad) = metric
            .evaluate_with_sensitivities(&residuals, &sensitivities)
            .expect("GaussianNll should support gradient evaluation");

        for (i, r) in residuals.iter().enumerate() {
            assert!((grad[i] - r / variance).abs() < 1e-10);
        }
    }

    #[cfg(not(feature = "cranelift-backend"))]
    #[test]
    fn diffsol_cost_gradient_matches_finite_difference() {
        let problem = build_logistic_problem(DiffsolBackend::Dense);
        let params = [1.1_f64, 0.9_f64];

        // Use the public API to obtain the analytical gradient
        let (cost, grad) = problem
            .evaluate_with_gradient(&params)
            .expect("cost with gradient calculation failed");

        assert!(cost.is_finite());
        let grad_vec = grad.expect("gradient should be present");
        assert_eq!(grad_vec.len(), params.len());

        let eps = 1e-5_f64;

        // Compare against finite-difference approximation of problem.evaluate
        #[allow(clippy::needless_range_loop)]
        for i in 0..params.len() {
            let mut params_fd = params;
            let fd = finite_difference(&mut params_fd, i, eps, |p| {
                problem
                    .evaluate(p)
                    .expect("finite-difference evaluation failed")
            });

            let g = grad_vec[i];
            let diff = (fd - g).abs();
            assert!(
                diff < 1e-6,
                "gradient mismatch for param {i}: fd={fd} grad={g} diff={diff}"
            );
        }
    }

    #[test]
    fn test_cache_reuses_simulator() {
        let problem = build_logistic_problem(DiffsolBackend::Dense);
        let params1 = [1.0_f64, 1.0_f64];
        let params2 = [1.1_f64, 0.9_f64];

        // First evaluation - should build simulator
        let cost1 = problem.evaluate(&params1).expect("First evaluation failed");
        assert!(cost1.is_finite());

        // Second evaluation with different params - should reuse simulator
        let cost2 = problem
            .evaluate(&params2)
            .expect("Second evaluation failed");
        assert!(cost2.is_finite());

        // Third evaluation with same params as second - should reuse and skip set_params
        let cost3 = problem.evaluate(&params2).expect("Third evaluation failed");
        assert!(
            (cost2 - cost3).abs() < 1e-10,
            "Same params should give same cost"
        );

        // Verify cache contains entry
        SIMULATOR_CACHE.with(|cache| {
            assert!(
                cache.borrow().contains_key(&problem.id),
                "Cache should contain problem"
            );
        });
    }

    #[test]
    fn test_cache_per_thread() {
        use std::thread;

        let problem = build_logistic_problem(DiffsolBackend::Dense);
        let params = [1.0_f64, 1.0_f64];

        // Evaluate in main thread
        problem
            .evaluate(&params)
            .expect("Main thread evaluation failed");

        let main_thread_has_cache =
            SIMULATOR_CACHE.with(|cache| cache.borrow().contains_key(&problem.id));
        assert!(main_thread_has_cache, "Main thread should have cache entry");

        // Spawn a new thread and verify it has separate cache
        let handle = thread::spawn(move || {
            SIMULATOR_CACHE.with(|cache| cache.borrow().contains_key(&problem.id))
        });

        let other_thread_has_cache = handle.join().expect("Thread join failed");
        assert!(
            !other_thread_has_cache,
            "Other thread should not have cache entry initially"
        );
    }

    #[test]
    fn test_cache_cleanup_on_drop() {
        let problem_id = {
            let problem = build_logistic_problem(DiffsolBackend::Dense);
            let params = [1.0_f64, 1.0_f64];

            problem.evaluate(&params).expect("Evaluation failed");

            // Verify cache has entry
            SIMULATOR_CACHE.with(|cache| {
                assert!(
                    cache.borrow().contains_key(&problem.id),
                    "Cache should contain problem"
                );
            });

            problem.id
        }; // problem dropped here

        // Verify cache was cleaned up
        SIMULATOR_CACHE.with(|cache| {
            assert!(
                !cache.borrow().contains_key(&problem_id),
                "Cache should not contain dropped problem"
            );
        });
    }

    #[test]
    fn test_multiple_objectives_different_caches() {
        let problem1 = build_logistic_problem(DiffsolBackend::Dense);
        let problem2 = build_logistic_problem(DiffsolBackend::Dense);
        let params = [1.0_f64, 1.0_f64];

        // Verify different IDs
        assert_ne!(
            problem1.id, problem2.id,
            "Different problems should have different IDs"
        );

        // Evaluate both
        problem1
            .evaluate(&params)
            .expect("Problem 1 evaluation failed");
        problem2
            .evaluate(&params)
            .expect("Problem 2 evaluation failed");

        // Verify both have cache entries
        SIMULATOR_CACHE.with(|cache| {
            let cache_map = cache.borrow();
            assert!(
                cache_map.contains_key(&problem1.id),
                "Cache should contain problem1"
            );
            assert!(
                cache_map.contains_key(&problem2.id),
                "Cache should contain problem2"
            );
            assert_eq!(cache_map.len(), 2, "Cache should have exactly 2 entries");
        });
    }
}
