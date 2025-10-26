use super::DiffsolConfig;
use diffsol::{
    DiffSl, MatrixCommon, NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverProblem,
};
use nalgebra::DMatrix;

use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

type M = diffsol::NalgebraMat<f64>;
type V = nalgebra::DVector<f64>;
type LS = diffsol::NalgebraLU<f64>;

#[cfg(not(feature = "diffsol-llvm"))]
type CG = diffsol::CraneliftJitModule;
#[cfg(feature = "diffsol-llvm")]
type CG = diffsol::LlvmModule;

type Eqn = DiffSl<M, CG>;

thread_local! {
    static PROBLEM_CACHE: RefCell<HashMap<usize, OdeSolverProblem<Eqn>>> = RefCell::new(HashMap::new());
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
        problem: OdeSolverProblem<Eqn>,
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

    fn build_problem(&self) -> Result<OdeSolverProblem<Eqn>, String> {
        let builder = OdeBuilder::<M>::new()
            .atol([self.config.atol])
            .rtol(self.config.rtol);

        builder
            .build_from_diffsl(&self.dsl)
            .map_err(|e| format!("Failed to build ODE model: {}", e))
    }

    fn seed_initial_problem(&self, problem: OdeSolverProblem<Eqn>) {
        let id = self.id;
        PROBLEM_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.insert(id, problem);
        });
    }

    fn with_thread_local_problem<F, R>(&self, mut f: F) -> Result<R, String>
    where
        F: FnMut(&mut OdeSolverProblem<Eqn>) -> Result<R, String>,
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
        problem: &mut OdeSolverProblem<Eqn>,
        params: &[f64],
        data: &DMatrix<f64>,
        t_span: &[f64],
    ) -> Result<f64, String> {
        let param_vec = NalgebraVec::from(V::from_vec(params.to_vec()));
        problem.eqn_mut().set_params(&param_vec);

        let solver = problem
            .bdf::<LS>()
            .map_err(|e| format!("Failed to create BDF solver: {}", e))?;

        let mut solver = solver;

        let Ok(solution) = solver.solve_dense(t_span) else {
            return Ok(1e5);
        };

        let sol_rows = solution.nrows();
        let sol_cols = solution.ncols();
        let data_rows = data.nrows();
        let data_cols = data.ncols();

        if sol_rows == data_rows && sol_cols == data_cols {
            let mut sum_sq = 0.0;
            for j in 0..sol_rows {
                for i in 0..sol_cols {
                    let diff = solution[(j, i)] - data[(j, i)];
                    sum_sq += diff * diff;
                }
            }
            Ok(sum_sq)
        } else if sol_rows == data_cols && sol_cols == data_rows {
            let mut sum_sq = 0.0;
            for j in 0..sol_rows {
                for i in 0..sol_cols {
                    let diff = solution[(j, i)] - data[(i, j)];
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
