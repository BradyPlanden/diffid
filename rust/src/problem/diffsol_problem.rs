use super::DiffsolConfig;
use diffsol::{
    DiffSl, MatrixCommon, NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod,
    OdeSolverProblem,
};
use nalgebra::DMatrix;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

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
    dsl: String,
    config: DiffsolConfig,
    pool: Arc<Mutex<Vec<OdeSolverProblem<Eqn>>>>,
    data: DMatrix<f64>,
    t_span: Vec<f64>,
}

impl DiffsolCost {
    pub fn new(
        problem: OdeSolverProblem<Eqn>,
        dsl: String,
        config: DiffsolConfig,
        data: DMatrix<f64>,
        t_span: Vec<f64>,
    ) -> Self {
        Self {
            dsl,
            config,
            pool: Arc::new(Mutex::new(vec![problem])),
            data,
            t_span,
        }
    }

    fn build_problem(&self) -> Result<OdeSolverProblem<Eqn>, String> {
        let builder = OdeBuilder::<M>::new()
            .atol([self.config.atol])
            .rtol(self.config.rtol);

        builder
            .build_from_diffsl(&self.dsl)
            .map_err(|e| format!("Failed to build ODE model: {}", e))
    }

    fn acquire_problem(&self) -> Result<OdeSolverProblem<Eqn>, String> {
        if let Some(problem) = self
            .pool
            .lock()
            .map_err(|e| format!("Mutex lock error: {}", e))?
            .pop()
        {
            return Ok(problem);
        }

        self.build_problem()
    }

    fn release_problem(&self, problem: OdeSolverProblem<Eqn>) -> Result<(), String> {
        self.pool
            .lock()
            .map_err(|e| format!("Mutex lock error: {}", e))?
            .push(problem);
        Ok(())
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
        let mut problem = self.acquire_problem()?;
        let result = Self::evaluate_with_problem(&mut problem, params, &self.data, &self.t_span);
        let release_result = self.release_problem(problem);
        release_result?;
        result
    }

    pub fn evaluate_population(&self, params: &[&[f64]]) -> Vec<Result<f64, String>> {
        params
            .par_iter()
            .map(|param| -> Result<f64, String> {
                let mut problem = self.acquire_problem()?;
                let result = Self::evaluate_with_problem(&mut problem, *param, &self.data, &self.t_span);
                self.release_problem(problem)?;
                result
            })
            .collect()
    }
}
