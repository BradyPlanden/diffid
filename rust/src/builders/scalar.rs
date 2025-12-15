use super::{ParameterSet, ProblemBuilderError};
use crate::optimisers::{NelderMead, Optimiser};
use crate::prelude::{ParameterSpec, Problem};
use crate::problem::{NoFunction, NoGradient, ScalarObjective};

#[derive(Clone)]
pub struct ScalarProblemBuilder<F = NoFunction, G = NoGradient, Opt = NelderMead> {
    f: F,
    gradient: G,
    parameters: ParameterSet,
    optimiser: Opt,
}

/// Initialises with empty function and gradient
impl ScalarProblemBuilder<NoFunction, NoGradient, NelderMead> {
    pub fn new() -> Self {
        Self {
            f: NoFunction,
            gradient: NoGradient,
            parameters: ParameterSet::default(),
            optimiser: NelderMead::new(),
        }
    }
}

/// Methods which are not state dependent
impl<F, G, Opt: Optimiser> ScalarProblemBuilder<F, G, Opt> {
    /// Update the optimiser from the default
    pub fn with_optimiser<NewOpt: Optimiser>(
        self,
        opt: NewOpt,
    ) -> ScalarProblemBuilder<F, G, NewOpt> {
        ScalarProblemBuilder {
            f: self.f,
            gradient: self.gradient,
            parameters: self.parameters,
            optimiser: opt,
        }
    }

    /// Add a parameter to the problem
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        initial: f64,
        bounds: Option<(f64, f64)>,
    ) -> Self {
        self.parameters
            .push(ParameterSpec::new(name, initial, bounds));
        self
    }
}

impl Default for ScalarProblemBuilder<NoFunction, NoGradient, NelderMead> {
    fn default() -> Self {
        Self::new()
    }
}

/// Add a function to a builder state with `NoFunction`
impl<G, Opt: Optimiser> ScalarProblemBuilder<NoFunction, G, Opt> {
    /// Stores the callable objective function
    pub fn with_function<F>(self, f: F) -> ScalarProblemBuilder<F, G, Opt>
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

impl<F, Opt: Optimiser> ScalarProblemBuilder<F, NoGradient, Opt> {
    /// Store the gradient objective function
    pub fn with_gradient<G>(self, gradient: G) -> ScalarProblemBuilder<F, G, Opt>
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
impl<F, Opt: Optimiser> ScalarProblemBuilder<F, NoGradient, Opt>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    Opt: Optimiser,
{
    /// Build the problem
    pub fn build(self) -> Result<Problem<ScalarObjective<F>, Opt>, ProblemBuilderError> {
        // Build objective
        let objective = ScalarObjective::new(self.f);

        // Build problem
        Ok(Problem::new(objective, self.parameters, self.optimiser))
    }
}

/// Build with gradient
impl<F, G, Opt: Optimiser> ScalarProblemBuilder<F, G, Opt>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    G: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    Opt: Optimiser,
{
    pub fn build(self) -> Result<Problem<ScalarObjective<F, G>, Opt>, ProblemBuilderError> {
        // Build objective
        let objective = ScalarObjective::with_gradient(self.f, self.gradient);
        Ok(Problem::new(objective, self.parameters, self.optimiser))
    }
}
