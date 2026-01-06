use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use std::time::Duration;

use chronopt_core::prelude::OptimisationResults;

#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::{PyNestedSamples, PySamples};

/// Request to evaluate objective function at specific points.
///
/// This is returned by `ask()` when the optimiser/sampler needs function
/// evaluations. Call `tell()` with the results after evaluation.
///
/// Examples
/// --------
/// >>> state = optimiser.init(problem, initial=[1.0, 2.0])
/// >>> result = state.ask()
/// >>> if isinstance(result, chronopt.Evaluate):
/// ...     values = [problem.evaluate(pt) for pt in result.points]
/// ...     state.tell(values)
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "Evaluate")]
#[derive(Clone)]
pub struct PyEvaluate {
    #[pyo3(get)]
    pub points: Vec<Vec<f64>>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyEvaluate {
    #[new]
    fn new(points: Vec<Vec<f64>>) -> Self {
        Self { points }
    }

    fn __repr__(&self) -> String {
        format!("Evaluate(points={} point(s))", self.points.len())
    }

    fn __str__(&self) -> String {
        format!("Evaluate {} points", self.points.len())
    }
}

/// Optimization/sampling is complete with final results.
///
/// This is returned by `ask()` when the algorithm has terminated.
/// Access the results via the `result` attribute.
///
/// Examples
/// --------
/// >>> while True:
/// ...     result = state.ask()
/// ...     if isinstance(result, chronopt.Done):
/// ...         print(f"Optimization complete: {result.result}")
/// ...         break
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "Done")]
pub struct PyDone {
    #[pyo3(get)]
    result: Py<PyAny>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyDone {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let result_repr = self.result.bind(py).repr()?.to_string();
        Ok(format!("Done(result={})", result_repr))
    }

    fn __str__(&self) -> String {
        "Done".to_string()
    }
}

impl PyDone {
    /// Create a Done variant with OptimisationResults
    pub fn with_optimisation_results(py: Python<'_>, results: OptimisationResults) -> Self {
        let py_results = PyOptimisationResults { inner: results };
        Self {
            result: Py::new(py, py_results).unwrap().into_any(),
        }
    }

    /// Create a Done variant with Samples (MCMC)
    pub fn with_samples(py: Python<'_>, samples: PySamples) -> Self {
        Self {
            result: Py::new(py, samples).unwrap().into_any(),
        }
    }

    /// Create a Done variant with NestedSamples
    pub fn with_nested_samples(py: Python<'_>, samples: PyNestedSamples) -> Self {
        Self {
            result: Py::new(py, samples).unwrap().into_any(),
        }
    }
}

/// Container for optimiser outputs and diagnostic metadata.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "OptimisationResults")]
pub struct PyOptimisationResults {
    pub(crate) inner: OptimisationResults,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyOptimisationResults {
    /// Decision vector corresponding to the best-found objective value.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Best parameter vector as a NumPy array
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.x.to_pyarray(py)
    }

    /// Objective value evaluated at `x`.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    /// Number of iterations performed by the optimiser.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Total number of objective function evaluations.
    #[getter]
    fn evaluations(&self) -> usize {
        self.inner.evaluations
    }

    /// Total number of objective function evaluations.
    #[getter]
    fn time(&self) -> Duration {
        self.inner.time
    }

    /// Whether the run satisfied its convergence criteria.
    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    /// Human-readable status message summarising the termination state.
    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }

    /// Structured termination flag describing why the run ended.
    #[getter]
    fn termination_reason(&self) -> String {
        self.inner.termination.to_string()
    }

    /// Simplex vertices at termination, when provided by the optimiser.
    #[getter]
    fn final_simplex(&self) -> Vec<Vec<f64>> {
        self.inner.final_simplex.clone()
    }

    /// Objective values corresponding to `final_simplex`.
    #[getter]
    fn final_simplex_values(&self) -> Vec<f64> {
        self.inner.final_simplex_values.clone()
    }

    /// Estimated covariance of the search distribution, if available.
    #[getter]
    fn covariance(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.covariance.clone()
    }

    /// Render a concise summary of the optimisation outcome.
    fn __repr__(&self) -> String {
        format!(
            "OptimisationResults(x={:?}, value={:.6}, iterations={}, evaluations={}, time={:?}, success={}, reason={})",
            self.inner.x,
            self.inner.value,
            self.inner.iterations,
            self.inner.evaluations,
            self.inner.time,
            self.inner.success,
            self.inner.message
        )
    }

    /// Return a human-readable summary of the result.
    fn __str__(&self) -> String {
        if self.inner.success {
            format!(
                "Success: f(x) = {:.6} after {} iterations",
                self.inner.value, self.inner.iterations
            )
        } else {
            format!(
                "Terminated: {} after {} iterations",
                self.inner.message, self.inner.iterations
            )
        }
    }

    /// Return truthiness based on optimization success.
    ///
    /// Allows using `if result:` instead of `if result.success:`.
    fn __bool__(&self) -> bool {
        self.inner.success
    }
}
