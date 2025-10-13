// use pyo3::prelude::*;
// use std::collections::HashMap;

// // Builder pattern for the optimisation problem
// #[pyclass]
// pub struct Builder {
//     callables: Vec<PyObject>,
//     config: HashMap<String, f64>,
// }

// #[pymethods]
// impl Builder {
//     fn add_callable(slf: Py<Self>, py: Python, obj: PyObject) -> PyResult<Py<Self>> {
//         let mut builder = slf.borrow_mut(py);
//         // callable verification
//         if !obj.bind(py).is_callable() {
//             return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
//                 "Object must be a callable",
//             ));
//         }
//         builder.callables.push(obj);
//         drop(builder);
//         Ok(slf)
//     }

//     fn build(&self, py: Python) -> PyResult<Problem> {
//         if self.callables.is_empty() {
//             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
//                 "At least one callable must be provided",
//             ));
//         }

//         Ok(Problem {
//             objective: self.callables[0].clone_ref(py), // clone zero index?
//             config: self.config.clone(),
//         })
//     }
// }

// // Problem factory for creating builders
// #[pyclass]
// pub struct SimpleProblem;

// #[pymethods]
// impl SimpleProblem {
//     fn __call__(&self) -> Builder {
//         Builder {
//             callables: Vec::new(),
//             config: HashMap::new(),
//         }
//     }
// }

// // Main API Entry
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

// // Problem class
// #[pyclass]
// pub struct Problem {
//     objective: PyObject,
//     config: HashMap<String, f64>,
// }

// #[pymethods]
// impl Problem {
//     pub fn evaluate(&self, py: Python, x: Vec<f64>) -> PyResult<f64> {
//         let result = self.objective.call1(py, (x,))?;
//         result.extract(py)
//     }
// }

// // Initial optimiser
// #[pyclass]
// pub struct NelderMead {
//     problem: Py<Problem>,
//     max_iter: usize,
//     threshold: f64,
// }

// #[pymethods]
// impl NelderMead {
//     #[new]
//     fn new(problem: Py<Problem>) -> Self {
//         Self {
//             problem,
//             max_iter: 1000,
//             threshold: 1e-6,
//         }
//     }

//     fn run(&self, py: Python) -> PyResult<OptimisationResults> {
//         let problem = self.problem.borrow(py);

//         // Simplified Nelder-Mead for testing
//         let mut x = vec![0.0, 0.0];
//         let mut best_val = problem.evaluate(py, x.clone())?;
//         let mut iterations = 0;

//         for i in 0..self.max_iter {
//             iterations += 1;

//             // Simplified step
//             let perturbation = 0.1 / (i as f64 + 1.0);
//             let mut improved = false;

//             for j in 0..x.len() {
//                 let mut x_new = x.clone();
//                 x_new[j] += perturbation;

//                 let val = problem.evaluate(py, x_new.clone())?;
//                 if val < best_val {
//                     x = x_new;
//                     best_val = val;
//                     improved = true;
//                 }
//             }

//             if !improved && perturbation < self.threshold {
//                 break;
//             }
//         }
//         Ok(OptimisationResults {
//             x,
//             fun: best_val,
//             nit: iterations,
//             success: true,
//         })
//     }
// }

// // Results object
// #[pyclass]
// pub struct OptimisationResults {
//     #[pyo3(get)]
//     x: Vec<f64>,
//     #[pyo3(get)]
//     fun: f64,
//     #[pyo3(get)]
//     nit: usize,
//     #[pyo3(get)]
//     success: bool,
// }

// #[pymethods]
// impl OptimisationResults {
//     fn __repr__(&self) -> String {
//         format!(
//             "OptimizationResults(x={:?}, fun={:.6}, nit={}, success={}))",
//             self.x, self.fun, self.nit, self.success
//         )
//     }
// }

// // Python module definition
// #[pymodule]
// fn chronopt(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_class::<problem::Builder>()?;
//     m.add_class::<problem::SimpleProblem>()?;
//     m.add_class::<problem::BuilderFactory>()?;
//     m.add_class::<problem::Problem>()?;
//     m.add_class::<optimisers::NelderMead>()?;
//     m.add_class::<optimisers::OptimisationResults>()?;
//     Ok(())
// }
