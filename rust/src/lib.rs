pub mod builders;
pub mod common;
pub mod cost;
pub mod optimisers;
pub mod problem;
pub mod sampler;

// Convenience re-exports so users can `use chronopt::prelude::*;`
pub mod prelude {
    pub use crate::builders::{
        DiffsolConfig, DiffsolProblemBuilder, ScalarProblemBuilder, VectorProblemBuilder,
    };
    pub use crate::common::{AskResult, Bounds, TellError, Unbounded};
    pub use crate::optimisers::{Adam, NelderMead, OptimisationResults, Optimiser, CMAES};
    pub use crate::problem::{Objective, ParameterSet, ParameterSpec, Problem};
    pub use crate::sampler::{
        DynamicNestedSampler, GradientSampler, MetropolisHastings, MetropolisHastingsState,
        NestedSample, NestedSamples, Sampler, Samples, SamplingResults, ScalarSampler,
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
        let result = optimiser.run(
            |x| problem.evaluate(x),
            vec![1.0, 1.0],
            Bounds::unbounded(2),
        );

        assert!(result.success);
        assert!(
            result.value < 0.01,
            "Expected fun < 0.01, but got: {}",
            result.value
        );
    }
}
