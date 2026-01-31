use super::{ParameterSet, ProblemBuilderError};
use crate::cost::{CostMetric, SumSquaredError};
use crate::optimisers::Optimiser;
use crate::prelude::{ParameterSpec, Problem};
use crate::problem::{ParameterRange, VectorFn, VectorObjective};
use std::sync::Arc;

#[must_use]
#[derive(Clone)]
pub struct VectorProblemBuilder {
    function: Option<VectorFn>,
    data: Option<Vec<f64>>,
    costs: Vec<Arc<dyn CostMetric>>,
    parameters: ParameterSet,
    optimiser: Optimiser,
}

impl Default for VectorProblemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorProblemBuilder {
    pub fn new() -> Self {
        Self {
            function: None,
            data: None,
            costs: Vec::new(),
            parameters: ParameterSet::default(),
            optimiser: Optimiser::default(),
        }
    }
}

impl VectorProblemBuilder {
    /// Stores the callable objective function
    pub fn with_function<F>(mut self, f: F) -> Self
    where
        F: Fn(&[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + 'static,
    {
        self.function = Some(Arc::new(f));
        self
    }

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

    /// Supplies observed data used to fit the differential model.
    pub fn with_data(mut self, data: Vec<f64>) -> Self {
        self.data = Some(data);
        self
    }

    /// Removes any previously supplied observed data and associated time span.
    pub fn remove_data(mut self) -> Self {
        self.data = None;
        self
    }

    /// Adds a cost used to compare model outputs against observed data.
    pub fn with_cost<M>(mut self, cost: M) -> Self
    where
        M: CostMetric + 'static,
    {
        self.costs.push(Arc::new(cost));
        self
    }

    /// Directly add the cost metric from a trait object.
    pub fn with_cost_arc(mut self, cost: Arc<dyn CostMetric>) -> Self {
        self.costs.push(cost);
        self
    }

    /// Resets the cost metric to the default sum of squared errors.
    pub fn remove_costs(mut self) -> Self {
        self.costs.clear();
        self
    }

    /// Build the problem
    pub fn build(self) -> Result<Problem<VectorObjective>, ProblemBuilderError> {
        // Default costs
        let mut costs = self.costs;
        if costs.is_empty() {
            costs.push(Arc::new(SumSquaredError::default()));
        }

        // Unpack attributes
        let function = self.function.ok_or(ProblemBuilderError::MissingVectorFn)?;
        let data = self.data.ok_or(ProblemBuilderError::MissingData)?;

        // Build objective
        let objective = VectorObjective::new(function, data, costs)?;

        // Build problem
        Ok(Problem::new(objective, self.parameters))
    }
}
