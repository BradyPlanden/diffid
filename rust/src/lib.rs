pub mod cost;
pub mod optimisers;
pub mod problem;
pub mod samplers;

// Convenience re-exports so users can `use chronopt::prelude::*;`
pub mod prelude {
    pub use crate::optimisers::{
        NelderMead, OptimisationResults, Optimiser, WithMaxIter, WithPatience, WithSigma0,
        WithThreshold, CMAES,
    };
    pub use crate::problem::builders::{BuilderOptimiserExt, BuilderParameterExt};
    pub use crate::problem::{Builder, DiffsolBuilder, ParameterSet, ParameterSpec, Problem};
    pub use crate::samplers::{MetropolisHastings, Sampler, Samples};
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    #[test]
    fn test_simple_optimisation() {
        let problem = Builder::new()
            .with_objective(|x: &[f64]| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();

        let optimiser = NelderMead::new().with_max_iter(500).with_sigma0(0.4);
        let result = optimiser.run(&problem, vec![1.0, 1.0]);

        assert!(result.success);
        assert!(
            result.fun < 0.01,
            "Expected fun < 0.01, but got: {}",
            result.fun
        );
    }

    #[test]
    fn test_diffsol_builder() {
        let dsl = r#"
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"#;

        let t_span: Vec<f64> = (0..5).map(|i| i as f64 * 0.1).collect();
        let data_values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let data = DMatrix::from_vec(5, 2, {
            let mut columns = Vec::with_capacity(10);
            columns.extend_from_slice(&t_span);
            columns.extend_from_slice(&data_values);
            columns
        });
        let config = HashMap::from([("rtol".to_string(), 1e-6)]);

        let builder = DiffsolBuilder::new()
            .with_diffsl(dsl.to_string())
            .with_data(data)
            .with_config(config)
            .with_parameter(ParameterSpec::new("r", 1.0, None))
            .with_parameter(ParameterSpec::new("k", 1.0, None));

        let problem = builder.build().unwrap();

        // Test that we can evaluate the problem
        let x0 = vec![1.0, 1.0]; // r, k parameters
        let cost = problem.evaluate(&x0);

        // Cost should be finite and non-negative
        assert!(
            cost.clone().unwrap().is_finite(),
            "Cost should be finite, got: {}",
            cost.unwrap()
        );
        assert!(
            cost.clone().unwrap() >= 0.0,
            "Cost should be non-negative, got: {}",
            cost.unwrap()
        );

        // Test optimization
        let result = problem.optimize(Some(x0), None);
        assert!(result.success);
        assert!(result.fun.is_finite());
    }
}
