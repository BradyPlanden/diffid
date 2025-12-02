use super::{DiffsolBackend, DiffsolConfig};
use crate::cost::CostMetric;
use diffsol::ode_solver::sensitivities::SensitivitiesOdeSolverMethod;
use diffsol::op::Op;
use diffsol::{
    DiffSl, FaerSparseLU, FaerSparseMat, FaerVec, Matrix, MatrixCommon, NalgebraLU, NalgebraMat,
    NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem, Vector,
};
use nalgebra::DMatrix;

use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
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
type DenseProblem = OdeSolverProblem<DenseEqn>;
type SparseProblem = OdeSolverProblem<SparseEqn>;

type DenseVector = NalgebraVec<f64>;
type SparseVector = FaerVec<f64>;
type DenseSolver = NalgebraLU<f64>;
type SparseSolver = FaerSparseLU<f64>;

pub enum BackendProblem {
    Dense(Box<DenseProblem>),
    Sparse(Box<SparseProblem>),
}

thread_local! {
    static PROBLEM_CACHE: RefCell<HashMap<usize, BackendProblem>> = RefCell::new(HashMap::new());
}

static NEXT_DIFFSOL_PROBLEM_ID: AtomicUsize = AtomicUsize::new(1);

const FAILED_SOLVE_PENALTY: f64 = 1e5;

/// Solver for Diffsol problems maintaining per-thread cached ODE instances.
///
/// # Thread Safety
///
/// This type uses thread-local storage to maintain per-thread ODE solver
/// problem instances, enabling safe parallel evaluation without locks.
/// Each thread lazily initializes its own problem instance on first use.
pub struct DiffsolProblem {
    id: usize,
    dsl: String,
    config: DiffsolConfig,
    t_span: Vec<f64>,
    data: DMatrix<f64>,
    cost_metric: Vec<Arc<dyn CostMetric>>,
}

pub enum SimulationResult {
    Solution(DMatrix<f64>),
    SolutionWithSensitivities(DMatrix<f64>, Vec<DMatrix<f64>>),
    Penalty,
}

impl DiffsolProblem {
    pub fn new(
        diffsol_problem: BackendProblem,
        dsl: String,
        config: DiffsolConfig,
        t_span: Vec<f64>,
        data: DMatrix<f64>,
        cost_metric: Vec<Arc<dyn CostMetric>>,
    ) -> Self {
        let id = NEXT_DIFFSOL_PROBLEM_ID.fetch_add(1, Ordering::Relaxed);
        let chron_problem = Self {
            id,
            dsl,
            config,
            t_span,
            data,
            cost_metric,
        };
        chron_problem.seed_initial_problem(diffsol_problem);
        chron_problem
    }

    fn build_problem(&self) -> Result<BackendProblem, String> {
        match self.config.backend {
            DiffsolBackend::Dense => OdeBuilder::<NalgebraMat<f64>>::new()
                .atol([self.config.atol])
                .rtol(self.config.rtol)
                .build_from_diffsl(&self.dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(|problem| BackendProblem::Dense(Box::new(problem))),
            DiffsolBackend::Sparse => OdeBuilder::<FaerSparseMat<f64>>::new()
                .atol([self.config.atol])
                .rtol(self.config.rtol)
                .build_from_diffsl(&self.dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(|problem| BackendProblem::Sparse(Box::new(problem))),
        }
    }

    fn seed_initial_problem(&self, problem: BackendProblem) {
        let id = self.id;
        PROBLEM_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.insert(id, problem);
        });
    }

    fn with_thread_local_problem<F, R>(&self, mut f: F) -> Result<R, String>
    where
        F: FnMut(&mut BackendProblem) -> Result<R, String>,
    {
        // Create ProbeGuard for both unit tests (cfg(test)) and integration tests
        // The ProbeGuard will only be active if a probe is registered
        let _probe_guard = test_support::ProbeGuard::new();

        let id = self.id;
        PROBLEM_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Entry::Vacant(e) = cache.entry(id) {
                e.insert(self.build_problem()?);
            }
            let problem = cache
                .get_mut(&id)
                .expect("problem cache must contain entry after insertion");
            match catch_unwind(AssertUnwindSafe(|| f(problem))) {
                Ok(result) => result,
                Err(_) => {
                    let rebuilt = self.build_problem()?;
                    let entry = cache
                        .get_mut(&id)
                        .expect("problem cache must contain entry after rebuild");
                    *entry = rebuilt;
                    let problem = cache
                        .get_mut(&id)
                        .expect("problem cache must contain entry after rebuild");
                    f(problem)
                }
            }
        })
    }

    fn simulate_diffsol(
        &self,
        problem: &mut BackendProblem,
        params: &[f64],
    ) -> Result<SimulationResult, String> {
        match problem {
            BackendProblem::Dense(problem) => {
                let ctx = *problem.eqn().context();
                let params_vec = DenseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf::<DenseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                let solve_result =
                    catch_unwind(AssertUnwindSafe(|| solver.solve_dense(&self.t_span)));

                match solve_result {
                    Ok(Ok(solution)) => Ok(SimulationResult::Solution(Self::matrix_to_dmatrix(
                        &solution,
                    ))),
                    _ => Ok(SimulationResult::Penalty),
                }
            }
            BackendProblem::Sparse(problem) => {
                let ctx = *problem.eqn().context();
                let params_vec = SparseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf::<SparseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                let solve_result =
                    catch_unwind(AssertUnwindSafe(|| solver.solve_dense(&self.t_span)));

                match solve_result {
                    Ok(Ok(solution)) => Ok(SimulationResult::Solution(Self::matrix_to_dmatrix(
                        &solution,
                    ))),
                    _ => Ok(SimulationResult::Penalty),
                }
            }
        }
    }

    fn simulate_diffsol_with_grad(
        &self,
        problem: &mut BackendProblem,
        params: &[f64],
    ) -> Result<SimulationResult, String> {
        match problem {
            BackendProblem::Dense(problem) => {
                let ctx = *problem.eqn().context();
                let params_vec = DenseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf_sens::<DenseSolver>()
                    .map_err(|e| format!("Failed to create BDF sensitivities solver: {}", e))?;

                let solve_result = catch_unwind(AssertUnwindSafe(|| {
                    solver.solve_dense_sensitivities(&self.t_span)
                }));

                match solve_result {
                    Ok(Ok((solution, sensitivities))) => {
                        let solution_dm = Self::matrix_to_dmatrix(&solution);
                        let sensitivities_dm =
                            sensitivities.iter().map(Self::matrix_to_dmatrix).collect();
                        Ok(SimulationResult::SolutionWithSensitivities(
                            solution_dm,
                            sensitivities_dm,
                        ))
                    }
                    _ => Ok(SimulationResult::Penalty),
                }
            }
            BackendProblem::Sparse(problem) => {
                let ctx = *problem.eqn().context();
                let params_vec = SparseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf_sens::<SparseSolver>()
                    .map_err(|e| format!("Failed to create BDF sensitivities solver: {}", e))?;

                let solve_result = catch_unwind(AssertUnwindSafe(|| {
                    solver.solve_dense_sensitivities(&self.t_span)
                }));

                match solve_result {
                    Ok(Ok((solution, sensitivities))) => {
                        let solution_dm = Self::matrix_to_dmatrix(&solution);
                        let sensitivities_dm =
                            sensitivities.iter().map(Self::matrix_to_dmatrix).collect();
                        Ok(SimulationResult::SolutionWithSensitivities(
                            solution_dm,
                            sensitivities_dm,
                        ))
                    }
                    _ => Ok(SimulationResult::Penalty),
                }
            }
        }
    }

    /// Simulate the ODE model with the given parameters and return the solution.
    /// If gradient is true, the gradient of the solution with respect to the parameters is also returned.
    pub fn simulate(&self, params: &[f64], gradient: bool) -> Result<SimulationResult, String> {
        self.with_thread_local_problem(|problem| {
            if gradient {
                Self::simulate_diffsol_with_grad(self, problem, params)
            } else {
                Self::simulate_diffsol(self, problem, params)
            }
        })
    }

    pub fn simulate_population(&self, params: &[&[f64]]) -> Vec<Result<SimulationResult, String>> {
        if self.config.parallel {
            params
                .par_iter()
                .map(|param| {
                    self.with_thread_local_problem(|problem| {
                        Self::simulate_diffsol(self, problem, param)
                    })
                })
                .collect()
        } else {
            params
                .iter()
                .map(|param| {
                    self.with_thread_local_problem(|problem| {
                        Self::simulate_diffsol(self, problem, param)
                    })
                })
                .collect()
        }
    }

    pub fn failed_solve_penalty() -> f64 {
        FAILED_SOLVE_PENALTY
    }

    pub fn is_parallel(&self) -> bool {
        self.config.parallel
    }

    fn build_residuals<M>(&self, solution: &M) -> Result<Vec<f64>, String>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let sol_rows = solution.nrows();
        let sol_cols = solution.ncols();
        let (data_rows, data_cols) = self.data.shape();

        let (transpose_data, expected_rows, expected_cols) =
            match (sol_rows, sol_cols, data_rows, data_cols) {
                (sr, sc, dr, dc) if sr == dr && sc == dc => (false, sr, sc),
                (sr, sc, dr, dc) if sr == dc && sc == dr => (true, sr, sc),
                (sr, sc, dr, dc) => {
                    return Err(format!(
                        "Solution shape {}x{} does not match data shape {}x{}",
                        sr, sc, dr, dc
                    ))
                }
            };

        let mut residuals = Vec::with_capacity(expected_rows * expected_cols);

        if transpose_data {
            for col in 0..expected_cols {
                for row in 0..expected_rows {
                    residuals.push(solution[(row, col)] - self.data[(col, row)]);
                }
            }
        } else {
            for row in 0..expected_rows {
                for col in 0..expected_cols {
                    residuals.push(solution[(row, col)] - self.data[(row, col)]);
                }
            }
        }

        Ok(residuals)
    }

    fn calculate_cost<M>(&self, solution: &M) -> Result<f64, String>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let residuals = self.build_residuals(solution)?;
        let total_cost = self
            .cost_metric
            .iter()
            .map(|metric| metric.evaluate(&residuals))
            .sum();
        Ok(total_cost)
    }

    fn calculate_cost_with_grad<M>(
        &self,
        solution: &M,
        sensitivities: &[M],
    ) -> Result<(f64, Vec<f64>), String>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let residuals = self.build_residuals(solution)?;

        // No conversion needed - call generic method directly
        self.cost_metric
            .iter()
            .try_fold((0.0, Vec::new()), |(acc_cost, acc_grad), metric| {
                let (cost, grad) = metric
                    .evaluate_with_sensitivities_generic(&residuals, sensitivities)
                    .ok_or_else(|| {
                        format!(
                            "Cost metric '{}' does not support gradient evaluation",
                            metric.name()
                        )
                    })?;

                let new_grad = if acc_grad.is_empty() {
                    grad
                } else {
                    acc_grad
                        .iter()
                        .zip(grad.iter())
                        .map(|(a, b)| a + b)
                        .collect()
                };

                Ok((acc_cost + cost, new_grad))
            })
    }


    fn calculate_cost_with_grad_from_generic_matrix<M>(
        &self,
        solution: &M,
        sensitivities: &[M],
    ) -> Result<(f64, Vec<f64>), String>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let residuals = self.build_residuals(solution)?;

        let sensitivities_dm: Vec<DMatrix<f64>> =
            sensitivities.iter().map(Self::matrix_to_dmatrix).collect();

        self.cost_metric
            .iter()
            .try_fold((0.0, Vec::new()), |(acc_cost, acc_grad), metric| {
                let (cost, grad) = metric
                    .evaluate_with_sensitivities(&residuals, &sensitivities_dm)
                    .ok_or_else(|| {
                        format!(
                            "Cost metric '{}' does not support gradient evaluation",
                            metric.name()
                        )
                    })?;

                let new_grad = if acc_grad.is_empty() {
                    grad
                } else {
                    acc_grad
                        .iter()
                        .zip(grad.iter())
                        .map(|(a, b)| a + b)
                        .collect()
                };

                Ok((acc_cost + cost, new_grad))
            })
    }

    pub fn evaluate(&self, params: &[f64]) -> Result<f64, String> {
        self.with_thread_local_problem(|problem| match problem {
            BackendProblem::Dense(p) => {
                let ctx = *p.eqn().context();
                p.eqn_mut().set_params(&DenseVector::from_vec(params.to_vec(), ctx));

                let mut solver = p.bdf::<DenseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                catch_unwind(AssertUnwindSafe(|| solver.solve_dense(&self.t_span)))
                    .ok()
                    .and_then(|r| r.ok())
                    .ok_or_else(|| "Diffsol solve failed".to_string())
                    .and_then(|solution| self.calculate_cost(&solution))
            }
            BackendProblem::Sparse(p) => {
                let ctx = *p.eqn().context();
                p.eqn_mut().set_params(&SparseVector::from_vec(params.to_vec(), ctx));

                let mut solver = p.bdf::<SparseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                catch_unwind(AssertUnwindSafe(|| solver.solve_dense(&self.t_span)))
                    .ok()
                    .and_then(|r| r.ok())
                    .ok_or_else(|| "Diffsol solve failed".to_string())
                    .and_then(|solution| self.calculate_cost(&solution))
            }
        })
    }

    pub fn evaluate_with_gradient(&self, params: &[f64]) -> Result<(f64, Vec<f64>), String> {
        self.with_thread_local_problem(|problem| match problem {
            BackendProblem::Dense(p) => {
                let ctx = *p.eqn().context();
                p.eqn_mut().set_params(&DenseVector::from_vec(params.to_vec(), ctx));

                let mut solver = p.bdf_sens::<DenseSolver>()
                    .map_err(|e| format!("Failed to create BDF sensitivities solver: {}", e))?;

                catch_unwind(AssertUnwindSafe(|| solver.solve_dense_sensitivities(&self.t_span)))
                    .ok()
                    .and_then(|r| r.ok())
                    .ok_or_else(|| "Diffsol solve failed".to_string())
                    .and_then(|(solution, sensitivities)| {
                        self.calculate_cost_with_grad(&solution, &sensitivities)
                    })
            }
            BackendProblem::Sparse(p) => {
                let ctx = *p.eqn().context();
                p.eqn_mut().set_params(&SparseVector::from_vec(params.to_vec(), ctx));

                let mut solver = p.bdf_sens::<SparseSolver>()
                    .map_err(|e| format!("Failed to create BDF sensitivities solver: {}", e))?;

                catch_unwind(AssertUnwindSafe(|| solver.solve_dense_sensitivities(&self.t_span)))
                    .ok()
                    .and_then(|r| r.ok())
                    .ok_or_else(|| "Diffsol solve failed".to_string())
                    .and_then(|(solution, sensitivities)| {
                        self.calculate_cost_with_grad(&solution, &sensitivities)
                    })
            }
        })
    }

    pub fn evaluate_population(&self, params: &[&[f64]]) -> Vec<Result<f64, String>> {
        let eval_fn = |param: &&[f64]| {
            self.with_thread_local_problem(|problem| {
                self.evaluate_single_with_penalty(problem, param)
            })
        };

        if self.config.parallel {
            params.par_iter().map(eval_fn).collect()
        } else {
            params.iter().map(eval_fn).collect()
        }
    }

    // Helper to avoid code duplication in evaluate_population
    fn evaluate_single_with_penalty(
        &self,
        problem: &mut BackendProblem,
        params: &[f64],
    ) -> Result<f64, String> {
        let result = match problem {
            BackendProblem::Dense(p) => {
                let ctx = *p.eqn().context();
                p.eqn_mut().set_params(&DenseVector::from_vec(params.to_vec(), ctx));

                p.bdf::<DenseSolver>()
                    .ok()
                    .and_then(|mut solver| {
                        catch_unwind(AssertUnwindSafe(|| solver.solve_dense(&self.t_span)))
                            .ok()
                            .and_then(|r| r.ok())
                    })
                    .and_then(|solution| self.calculate_cost(&solution).ok())
            }
            BackendProblem::Sparse(p) => {
                let ctx = *p.eqn().context();
                p.eqn_mut().set_params(&SparseVector::from_vec(params.to_vec(), ctx));

                p.bdf::<SparseSolver>()
                    .ok()
                    .and_then(|mut solver| {
                        catch_unwind(AssertUnwindSafe(|| solver.solve_dense(&self.t_span)))
                            .ok()
                            .and_then(|r| r.ok())
                    })
                    .and_then(|solution| self.calculate_cost(&solution).ok())
            }
        };

        Ok(result.unwrap_or_else(|| Self::failed_solve_penalty()))
    }

    fn matrix_to_dmatrix<M>(matrix: &M) -> DMatrix<f64>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(matrix[(row, col)]);
            }
        }
        DMatrix::from_row_slice(rows, cols, &data)
    }
}

/// Clean-up for globally stored
/// PROBLEM_CACHE HashMap
impl Drop for DiffsolProblem {
    fn drop(&mut self) {
        let id = self.id;
        PROBLEM_CACHE.with(|cache| {
            cache.borrow_mut().remove(&id);
        });
    }
}

pub mod test_support {
    use super::*;
    use std::sync::{atomic::Ordering, Mutex};
    use std::time::Duration;

    #[derive(Clone)]
    pub struct ConcurrencyProbe {
        inner: std::sync::Arc<ProbeInner>,
    }

    struct ProbeInner {
        active: AtomicUsize,
        peak: AtomicUsize,
        sleep: Duration,
    }

    impl ConcurrencyProbe {
        pub fn new(sleep: Duration) -> Self {
            Self {
                inner: std::sync::Arc::new(ProbeInner {
                    active: AtomicUsize::new(0),
                    peak: AtomicUsize::new(0),
                    sleep,
                }),
            }
        }

        fn enter(&self) {
            let current = self.inner.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.inner.peak.fetch_max(current, Ordering::SeqCst);
            if !self.inner.sleep.is_zero() {
                std::thread::sleep(self.inner.sleep);
            }
        }

        fn exit(&self) {
            self.inner.active.fetch_sub(1, Ordering::SeqCst);
        }

        pub fn peak(&self) -> usize {
            self.inner.peak.load(Ordering::SeqCst)
        }
    }

    static PROBE_REGISTRY: Mutex<Option<ConcurrencyProbe>> = Mutex::new(None);

    fn set_probe_internal(probe: Option<ConcurrencyProbe>) {
        *PROBE_REGISTRY
            .lock()
            .expect("probe registry mutex poisoned") = probe;
    }

    pub struct ProbeInstall;

    impl ProbeInstall {
        pub fn new(probe: Option<ConcurrencyProbe>) -> Self {
            set_probe_internal(probe);
            Self
        }
    }

    impl Drop for ProbeInstall {
        fn drop(&mut self) {
            set_probe_internal(None);
        }
    }

    pub struct ProbeGuard(Option<ConcurrencyProbe>);

    impl Default for ProbeGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ProbeGuard {
        pub fn new() -> Self {
            let probe = PROBE_REGISTRY
                .lock()
                .expect("probe registry mutex poisoned")
                .clone();
            if let Some(probe) = probe {
                probe.enter();
                ProbeGuard(Some(probe))
            } else {
                ProbeGuard(None)
            }
        }
    }

    impl Drop for ProbeGuard {
        fn drop(&mut self) {
            if let Some(probe) = &self.0 {
                probe.exit();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{GaussianNll, SumSquaredError};

    fn build_logistic_problem(backend: DiffsolBackend) -> DiffsolProblem {
        let dsl = r#"
in = [r, k]
r { 1 }
k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"#;

        let t_span: Vec<f64> = (0..6).map(|i| i as f64 * 0.2).collect();
        let data_values: Vec<f64> = t_span.iter().map(|t| 0.1 * (*t).exp()).collect();
        let data = DMatrix::from_vec(t_span.len(), 1, data_values);

        let config = DiffsolConfig::default().with_backend(backend);

        let backend_problem = match config.backend {
            DiffsolBackend::Dense => OdeBuilder::<NalgebraMat<f64>>::new()
                .atol([config.atol])
                .rtol(config.rtol)
                .build_from_diffsl(dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(|problem| BackendProblem::Dense(Box::new(problem)))
                .expect("failed to build dense diffsol problem"),
            DiffsolBackend::Sparse => OdeBuilder::<FaerSparseMat<f64>>::new()
                .atol([config.atol])
                .rtol(config.rtol)
                .build_from_diffsl(dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(|problem| BackendProblem::Sparse(Box::new(problem)))
                .expect("failed to build sparse diffsol problem"),
        };

        DiffsolProblem::new(
            backend_problem,
            dsl.to_string(),
            config,
            t_span,
            data,
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
        let (cost, grad) = metric
            .evaluate_with_sensitivities(&residuals, &[])
            .expect("SumSquaredError should support gradient evaluation");
        assert_eq!(cost, 5.0);
        assert!(grad.is_empty());
    }

    #[test]
    fn test_gradient_dimensions_mismatch() {
        let metric = SumSquaredError::default();
        let residuals = vec![1.0, 2.0, 3.0];
        let wrong_size_sens = DMatrix::from_vec(2, 1, vec![0.5, 0.5]);

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
        let sensitivities: Vec<DMatrix<f64>> = (0..3)
            .map(|param_idx| {
                let mut m = DMatrix::zeros(3, 1);
                m[(param_idx, 0)] = 1.0;
                m
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
    fn diffsol_simulate_produces_sensitivities() {
        let problem = build_logistic_problem(DiffsolBackend::Dense);
        let params = [1.0_f64, 1.0_f64];

        let result = problem
            .simulate(&params, true)
            .expect("simulation with sensitivities failed");

        let (solution, sensitivities) = match result {
            SimulationResult::SolutionWithSensitivities(solution, sensitivities) => {
                (solution, sensitivities)
            }
            _ => panic!("expected solution with sensitivities"),
        };

        let (rows, cols) = solution.shape();
        assert_eq!(sensitivities.len(), params.len());
        for sens in &sensitivities {
            assert_eq!(sens.nrows(), rows);
            assert_eq!(sens.ncols(), cols);
        }

        let row = 0;
        let col = 0;
        let eps = 1e-5_f64;

        let mut params_fd = params;
        let fd = finite_difference(&mut params_fd, 0, eps, |p| {
            let result = problem
                .simulate(p, false)
                .expect("finite-difference simulation failed");
            let solution = match result {
                SimulationResult::Solution(solution) => solution,
                _ => panic!("expected solution without sensitivities"),
            };
            solution[(row, col)]
        });

        let sens_value = sensitivities[0][(row, col)];
        let diff = (fd - sens_value).abs();
        assert!(
            diff < 1e-6,
            "sensitivity mismatch: fd={} sens={} diff={}",
            fd,
            sens_value,
            diff
        );
    }

    #[cfg(not(feature = "cranelift-backend"))]
    #[test]
    fn diffsol_cost_gradient_matches_finite_difference() {
        let problem = build_logistic_problem(DiffsolBackend::Dense);
        let params = [1.1_f64, 0.9_f64];

        let result = problem
            .simulate(&params, true)
            .expect("simulation with sensitivities failed");

        let (solution, sensitivities) = match result {
            SimulationResult::SolutionWithSensitivities(solution, sensitivities) => {
                (solution, sensitivities)
            }
            _ => panic!("expected solution with sensitivities"),
        };

        let (cost, grad) = problem
            .calculate_cost_with_grad(&solution, &sensitivities)
            .expect("cost with gradient calculation failed");

        assert!(cost.is_finite());
        assert_eq!(grad.len(), params.len());

        let eps = 1e-5_f64;

        for i in 0..params.len() {
            let mut params_fd = params;
            let fd = finite_difference(&mut params_fd, i, eps, |p| {
                let result = problem
                    .simulate(p, false)
                    .expect("finite-difference simulation failed");
                let solution = match result {
                    SimulationResult::Solution(solution) => solution,
                    _ => panic!("expected solution without sensitivities"),
                };
                problem
                    .calculate_cost(&solution)
                    .expect("cost calculation failed")
            });

            let g = grad[i];
            let diff = (fd - g).abs();
            assert!(
                diff < 1e-6,
                "gradient mismatch for param {}: fd={} grad={} diff={}",
                i,
                fd,
                g,
                diff
            );
        }
    }
}
