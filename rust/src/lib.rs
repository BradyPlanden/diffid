pub mod optimisers;
pub mod problem;

// Convenience re-exports so users can `use chronopt::prelude::*;`
pub mod prelude {
    pub use crate::optimisers::{
        NelderMead, OptimisationResults, Optimiser, WithMaxIter, WithSigma0, WithThreshold,
    };
    pub use crate::problem::{Builder, DiffsolBuilder, Problem};
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

        let data = DMatrix::from_vec(5, 1, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let config = HashMap::from([("rtol".to_string(), 1e-6)]);
        let params = HashMap::from([("r".to_string(), 1.0), ("k".to_string(), 1.0)]);

        let builder = DiffsolBuilder::new()
            .add_diffsl(dsl.to_string())
            .add_data(data)
            .add_config(config)
            .add_params(params);

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
