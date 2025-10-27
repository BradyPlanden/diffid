use super::{DiffsolBackend, DiffsolConfig};
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
use std::sync::atomic::{AtomicUsize, Ordering};

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
    Dense(DenseProblem),
    Sparse(SparseProblem),
}

thread_local! {
    static PROBLEM_CACHE: RefCell<HashMap<usize, BackendProblem>> = RefCell::new(HashMap::new());
}

static NEXT_DIFFSOL_COST_ID: AtomicUsize = AtomicUsize::new(1);

/// Cost function for Diffsol problems
pub struct DiffsolCost {
    id: usize,
    dsl: String,
    config: DiffsolConfig,
    data: DMatrix<f64>,
    t_span: Vec<f64>,
}

/// Cost function for Diffsol problems.
///
/// # Thread Safety
///
/// This type uses thread-local storage to maintain per-thread ODE solver
/// problem instances, enabling safe parallel evaluation without locks.
/// Each thread lazily initializes its own problem instance on first use.
impl DiffsolCost {
    pub fn new(
        problem: BackendProblem,
        dsl: String,
        config: DiffsolConfig,
        data: DMatrix<f64>,
        t_span: Vec<f64>,
    ) -> Self {
        let id = NEXT_DIFFSOL_COST_ID.fetch_add(1, Ordering::Relaxed);
        let cost = Self {
            id,
            dsl,
            config,
            data,
            t_span,
        };
        cost.seed_initial_problem(problem);
        cost
    }

    fn build_problem(&self) -> Result<BackendProblem, String> {
        match self.config.backend {
            DiffsolBackend::Dense => OdeBuilder::<NalgebraMat<f64>>::new()
                .atol([self.config.atol])
                .rtol(self.config.rtol)
                .build_from_diffsl(&self.dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(BackendProblem::Dense),
            DiffsolBackend::Sparse => OdeBuilder::<FaerSparseMat<f64>>::new()
                .atol([self.config.atol])
                .rtol(self.config.rtol)
                .build_from_diffsl(&self.dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(BackendProblem::Sparse),
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
            f(problem)
        })
    }

    fn evaluate_with_problem(
        problem: &mut BackendProblem,
        params: &[f64],
        data: &DMatrix<f64>,
        t_span: &[f64],
    ) -> Result<f64, String> {
        match problem {
            BackendProblem::Dense(problem) => {
                let ctx = problem.eqn().context().clone();
                let params_vec = DenseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf::<DenseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                match solver.solve_dense(t_span) {
                    Ok(solution) => Self::matrix_squared_error(&solution, data),
                    Err(_) => Ok(1e5),
                }
            }
            BackendProblem::Sparse(problem) => {
                let ctx = problem.eqn().context().clone();
                let params_vec = SparseVector::from_vec(params.to_vec(), ctx);
                problem.eqn_mut().set_params(&params_vec);

                let mut solver = problem
                    .bdf::<SparseSolver>()
                    .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

                match solver.solve_dense(t_span) {
                    Ok(solution) => Self::matrix_squared_error(&solution, data),
                    Err(_) => Ok(1e5),
                }
            }
        }
    }

    fn matrix_squared_error<M>(solution: &M, data: &DMatrix<f64>) -> Result<f64, String>
    where
        M: Matrix + MatrixCommon + Index<(usize, usize), Output = f64>,
    {
        let sol_rows = solution.nrows() as usize;
        let sol_cols = solution.ncols() as usize;
        let data_rows = data.nrows();
        let data_cols = data.ncols();

        if sol_rows == data_rows && sol_cols == data_cols {
            let mut sum_sq = 0.0;
            for row in 0..sol_rows {
                for col in 0..sol_cols {
                    let diff = solution[(row, col)] - data[(row, col)];
                    sum_sq += diff * diff;
                }
            }
            Ok(sum_sq)
        } else if sol_rows == data_cols && sol_cols == data_rows {
            let mut sum_sq = 0.0;
            for row in 0..sol_rows {
                for col in 0..sol_cols {
                    let diff = solution[(row, col)] - data[(col, row)];
                    sum_sq += diff * diff;
                }
            }
            Ok(sum_sq)
        } else {
            Err(format!(
                "Solution shape {}x{} does not match data shape {}x{}",
                sol_rows, sol_cols, data_rows, data_cols
            ))
        }
    }

    /// Evaluate cost function (sum of squared differences)
    /// Falls back to a large penalty when the solver fails to converge.
    pub fn evaluate(&self, params: &[f64]) -> Result<f64, String> {
        self.with_thread_local_problem(|problem| {
            Self::evaluate_with_problem(problem, params, &self.data, &self.t_span)
        })
    }

    pub fn evaluate_population(&self, params: &[&[f64]]) -> Vec<Result<f64, String>> {
        params
            .par_iter()
            .map(|param| {
                self.with_thread_local_problem(|problem| {
                    Self::evaluate_with_problem(problem, param, &self.data, &self.t_span)
                })
            })
            .collect()
    }
}

/// Clean-up for globally stored
/// PROBLEM_CACHE HashMap
impl Drop for DiffsolCost {
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
