use std::collections::HashMap;

pub type ObjectiveFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

// Builder pattern for the optimisation problem
pub struct Builder {
    objective: Option<ObjectiveFn>,
    config: HashMap<String, f64>,
}
impl Builder {
    pub fn new() -> Self {
        Self {
            objective: None,
            config: HashMap::new(),
        }
    }

    pub fn with_objective<F>(mut self, f: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.objective = Some(Box::new(f));
        self
    }

    pub fn with_config(mut self, key: String, value: f64) -> Self {
        self.config.insert(key, value);
        self
    }

    pub fn build(self) -> Result<Problem, String> {
        match self.objective {
            Some(obj) => Ok(Problem {
                objective: obj,
                config: self.config,
            }),
            None => Err("At least one objective must be provide".to_string()),
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

// Problem factory for creating builders
// pub struct SimpleProblem;

// impl SimpleProblem {
//     fn __call__(&self) -> Builder {
//         Builder {
//             callables: Vec::new(),
//             config: HashMap::new(),
//         }
//     }
// }

// Main API Entry
// #[pyclass]
// pub struct BuilderFactory;

// #[pymethods]
// impl BuilderFactory {
//     #[new]
//     fn new() -> Self {
//         Self
//     }

//     #[getter]
//     fn SimpleProblem(&self) -> SimpleProblem {
//         SimpleProblem
//     }
// }

// Problem class
pub struct Problem {
    objective: ObjectiveFn,
    config: HashMap<String, f64>,
}

impl Problem {
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        (self.objective)(x)
    }

    pub fn get_config(&self, key: &str) -> Option<&f64> {
        self.config.get(key)
    }

    pub fn config(&self) -> &HashMap<String, f64> {
        &self.config
    }
}
