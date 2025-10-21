use diffsol::{DiffSl, MatrixCommon, NalgebraVec, OdeEquations, OdeSolverMethod, OdeSolverProblem};
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::sync::Mutex;

type M = diffsol::NalgebraMat<f64>;
type V = nalgebra::DVector<f64>;
type LS = diffsol::NalgebraLU<f64>;

#[cfg(not(feature = "diffsol-llvm"))]
type CG = diffsol::CraneliftJitModule;
#[cfg(feature = "diffsol-llvm")]
type CG = diffsol::LlvmModule;

type Eqn = DiffSl<M, CG>;

/// Cost function for Diffsol problems
pub struct DiffsolCost {
    problem: Mutex<OdeSolverProblem<Eqn>>,
    data: DMatrix<f64>,
    t_span: Vec<f64>,
    config: HashMap<String, f64>,
}

impl DiffsolCost {
    pub fn new(
        problem: OdeSolverProblem<Eqn>,
        data: DMatrix<f64>,
        t_span: Vec<f64>,
        config: HashMap<String, f64>,
    ) -> Self {
        Self {
            problem: Mutex::new(problem),
            data,
            t_span,
            config,
        }
    }

    /// Evaluate cost function (sum of squared differences)
    /// This is currently non-optimised, with the solution type
    /// NalgebraVec manually iterated over due to type conversion
    /// problems.
    pub fn evaluate(&self, params: &[f64]) -> Result<f64, String> {
        let mut problem = self
            .problem
            .lock()
            .map_err(|e| format!("Mutex lock error: {}", e))?;

        // Create NalgebraVec instead of DVector
        let param_vec = NalgebraVec::from(V::from_vec(params.to_vec()));
        problem.eqn_mut().set_params(&param_vec);

        let Ok(solution) = problem.bdf::<LS>().unwrap().solve_dense(&self.t_span) else {
            return Ok(1000.0);
        };

        let nrows = solution.nrows();
        let ncols = solution.ncols();

        let cost: f64 = (0..ncols)
            .map(|i| {
                let mut sum_sq = 0.0;
                for j in 0..nrows {
                    let diff = solution[(j, i)] - self.data[(j, i)];
                    sum_sq += diff * diff;
                }
                sum_sq
            })
            .sum();

        Ok(cost)
    }
}
