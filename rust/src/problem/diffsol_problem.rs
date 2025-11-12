use super::{DiffsolBackend, DiffsolConfig};
use crate::cost::CostMetric;
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

#[cfg(not(feature = "diffsol-llvm"))]
type CG = diffsol::CraneliftJitModule;
#[cfg(feature = "diffsol-llvm")]
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
    cost_metric: Arc<dyn CostMetric>,
}

pub enum SimulationResult {
    Solution(DMatrix<f64>),
    Penalty,
}

impl DiffsolProblem {
    pub fn new(
        problem: BackendProblem,
        dsl: String,
        config: DiffsolConfig,
        t_span: Vec<f64>,
        data: DMatrix<f64>,
        cost_metric: Arc<dyn CostMetric>,
    ) -> Self {
        let id = NEXT_DIFFSOL_PROBLEM_ID.fetch_add(1, Ordering::Relaxed);
        let solver = Self {
            id,
            dsl,
            config,
            t_span,
            data,
            cost_metric,
        };
        solver.seed_initial_problem(problem);
        solver
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
        #[cfg(test)]
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

    fn simulate_with_problem(
        problem: &mut BackendProblem,
        params: &[f64],
        t_span: &[f64],
    ) -> Result<SimulationResult, String> {
        match problem {
            BackendProblem::Dense(problem) => {
                let ctx = problem.eqn().context().clone();
                let params_vec = DenseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf::<DenseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                let solve_result = catch_unwind(AssertUnwindSafe(|| solver.solve_dense(t_span)));

                match solve_result {
                    Ok(Ok(solution)) => Ok(SimulationResult::Solution(Self::matrix_to_dmatrix(
                        &solution,
                    ))),
                    _ => Ok(SimulationResult::Penalty),
                }
            }
            BackendProblem::Sparse(problem) => {
                let ctx = problem.eqn().context().clone();
                let params_vec = SparseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf::<SparseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                let solve_result = catch_unwind(AssertUnwindSafe(|| solver.solve_dense(t_span)));

                match solve_result {
                    Ok(Ok(solution)) => Ok(SimulationResult::Solution(Self::matrix_to_dmatrix(
                        &solution,
                    ))),
                    _ => Ok(SimulationResult::Penalty),
                }
            }
        }
    }

    pub fn simulate(&self, params: &[f64]) -> Result<SimulationResult, String> {
        self.with_thread_local_problem(|problem| {
            Self::simulate_with_problem(problem, params, &self.t_span)
        })
    }

    pub fn simulate_population(&self, params: &[&[f64]]) -> Vec<Result<SimulationResult, String>> {
        if self.config.parallel {
            params
                .par_iter()
                .map(|param| {
                    self.with_thread_local_problem(|problem| {
                        Self::simulate_with_problem(problem, param, &self.t_span)
                    })
                })
                .collect()
        } else {
            params
                .iter()
                .map(|param| {
                    self.with_thread_local_problem(|problem| {
                        Self::simulate_with_problem(problem, param, &self.t_span)
                    })
                })
                .collect()
        }
    }

    pub fn failed_solve_penalty() -> f64 {
        FAILED_SOLVE_PENALTY
    }

    pub fn calculate_cost(&self, solution: &DMatrix<f64>) -> Result<f64, String> {
        let sol_rows = solution.nrows();
        let sol_cols = solution.ncols();
        let data_rows = self.data.nrows();
        let data_cols = self.data.ncols();

        let residuals = if sol_rows == data_rows && sol_cols == data_cols {
            let mut residuals = Vec::with_capacity(sol_rows * sol_cols);
            for row in 0..sol_rows {
                for col in 0..sol_cols {
                    residuals.push(solution[(row, col)] - self.data[(row, col)]);
                }
            }
            residuals
        } else if sol_rows == data_cols && sol_cols == data_rows {
            let mut residuals = Vec::with_capacity(sol_rows * sol_cols);
            for row in 0..sol_rows {
                for col in 0..sol_cols {
                    residuals.push(solution[(row, col)] - self.data[(col, row)]);
                }
            }
            residuals
        } else {
            return Err(format!(
                "Solution shape {}x{} does not match data shape {}x{}",
                sol_rows, sol_cols, data_rows, data_cols
            ));
        };

        Ok(self.cost_metric.evaluate(&residuals))
    }

    pub fn evaluate(&self, params: &[f64]) -> Result<f64, String> {
        match self.simulate(params)? {
            SimulationResult::Solution(solution) => self.calculate_cost(&solution),
            SimulationResult::Penalty => Ok(DiffsolProblem::failed_solve_penalty()),
        }
    }

    pub fn evaluate_population(&self, params: &[&[f64]]) -> Vec<Result<f64, String>> {
        let simulate_results = self.simulate_population(params);

        if self.config.parallel {
            simulate_results
                .into_par_iter()
                .map(|result| match result {
                    Ok(SimulationResult::Solution(solution)) => self.calculate_cost(&solution),
                    Ok(SimulationResult::Penalty) => Ok(DiffsolProblem::failed_solve_penalty()),
                    Err(err) => Err(err),
                })
                .collect()
        } else {
            simulate_results
                .into_iter()
                .map(|result| match result {
                    Ok(SimulationResult::Solution(solution)) => self.calculate_cost(&solution),
                    Ok(SimulationResult::Penalty) => Ok(DiffsolProblem::failed_solve_penalty()),
                    Err(err) => Err(err),
                })
                .collect()
        }
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

#[cfg(test)]
pub(crate) mod test_support {
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
