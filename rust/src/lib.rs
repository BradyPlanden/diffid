pub mod builders;
pub mod cost;
pub mod optimisers;
pub mod problem;
pub mod sampler;

// Convenience re-exports so users can `use chronopt::prelude::*;`
pub mod prelude {
    pub use crate::optimisers::{
        Adam, NelderMead, OptimisationResults, Optimiser, WithMaxIter, WithPatience, WithSigma0,
        WithThreshold, CMAES,
    };
    // pub use crate::builders::{BuilderOptimiserExt, BuilderParameterExt};
    pub use crate::builders::{
        DiffsolConfig, DiffsolProblemBuilder, ScalarProblemBuilder, VectorProblemBuilder,
    };
    pub use crate::problem::{Objective, ParameterSet, ParameterSpec, Problem};
    pub use crate::sampler::{
        DynamicNestedSampler, MetropolisHastings, NestedSample, NestedSamples, Sampler, Samples,
    };
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_simple_optimisation() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x[0].powi(2) + x[1].powi(2))
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
}
