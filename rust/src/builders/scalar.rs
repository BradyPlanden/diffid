use super::{ParameterSet, ProblemBuilderError};
use crate::optimisers::Optimiser;
use crate::prelude::{ParameterSpec, Problem};
use crate::problem::{NoFunction, NoGradient, ParameterRange, ScalarObjective};

#[derive(Clone)]
pub struct ScalarProblemBuilder<F = NoFunction, G = NoGradient> {
    f: F,
    gradient: G,
    parameters: ParameterSet,
    optimiser: Optimiser,
}

/// Initialises with empty function and gradient
impl ScalarProblemBuilder<NoFunction, NoGradient> {
    pub fn new() -> Self {
        Self {
            f: NoFunction,
            gradient: NoGradient,
            parameters: ParameterSet::default(),
            optimiser: Optimiser::default(),
        }
    }
}

/// Methods which are not state dependent
impl<F, G> ScalarProblemBuilder<F, G> {
    /// Update the optimiser from the default
    pub fn with_optimiser(mut self, opt: impl Into<Optimiser>) -> Self {
        self.optimiser = opt.into();
        self
    }

    /// Add a parameter to the problem
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        initial: f64,
        range: impl Into<ParameterRange>,
    ) -> Self {
        self.parameters
            .push(ParameterSpec::new(name, initial, range));
        self
    }
}

impl Default for ScalarProblemBuilder<NoFunction, NoGradient> {
    fn default() -> Self {
        Self::new()
    }
}

/// Add a function to a builder state with `NoFunction`
impl<G> ScalarProblemBuilder<NoFunction, G> {
    /// Stores the callable objective function
    pub fn with_function<F>(self, f: F) -> ScalarProblemBuilder<F, G>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        ScalarProblemBuilder {
            f,
            gradient: self.gradient,
            parameters: self.parameters,
            optimiser: self.optimiser,
        }
    }
}

impl<F> ScalarProblemBuilder<F, NoGradient> {
    /// Store the gradient objective function
    pub fn with_gradient<G>(self, gradient: G) -> ScalarProblemBuilder<F, G>
    where
        G: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    {
        ScalarProblemBuilder {
            f: self.f,
            gradient,
            parameters: self.parameters,
            optimiser: self.optimiser,
        }
    }
}

/// Build without gradient
impl<F> ScalarProblemBuilder<F, NoGradient>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
{
    /// Build the problem
    pub fn build(self) -> Result<Problem<ScalarObjective<F>>, ProblemBuilderError> {
        // Build objective
        let objective = ScalarObjective::new(self.f);

        // Build problem
        Ok(Problem::new(objective, self.parameters, self.optimiser))
    }
}

/// Build with gradient
impl<F, G> ScalarProblemBuilder<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    G: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
{
    pub fn build(self) -> Result<Problem<ScalarObjective<F, G>>, ProblemBuilderError> {
        // Build objective
        let objective = ScalarObjective::with_gradient(self.f, self.gradient);
        Ok(Problem::new(objective, self.parameters, self.optimiser))
    }
}
