use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

struct PyObjectiveFn {
    callable: PyObject,
}

// Wrapper for Python callable
impl PyObjectiveFn {
    fn call(&self, x: &[f64]) -> PyResult<f64> {
        Python::with_gil(|py| {
            let callable = self.callable.bind(py);
            callable.call1((x.to_vec(),))?.extract()
        })
    }
}

#[pyclass(name = "Builder")]
pub struct PyBuilder {
    inner: chronopt_core::problem::Builder,
    py_callable: Option<Arc<PyObjectiveFn>>,
}

#[pymethods]
impl PyBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: chronopt_core::problem::Builder::new(),
            py_callable: None,
        }
    }

    fn add_callable<'a>(
        mut slf: PyRefMut<'a, Self>,
        obj: PyObject,
        py: Python<'_>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if !obj.bind(py).is_callable() {
            return Err(PyTypeError::new_err("Object must be callable"));
        }

        let py_fn = Arc::new(PyObjectiveFn { callable: obj });
        let py_fn_clone = Arc::clone(&py_fn);

        slf.inner = std::mem::take(&mut slf.inner)
            .with_objective(move |x: &[f64]| py_fn_clone.call(x).unwrap_or(f64::INFINITY));

        slf.py_callable = Some(py_fn);
        Ok(slf)
    }

    fn with_config(mut slf: PyRefMut<'_, Self>, key: String, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_config(key, value);
        slf
    }

    fn build(&mut self) -> PyResult<PyProblem> {
        let inner = std::mem::take(&mut self.inner);
        match inner.build() {
            Ok(problem) => Ok(PyProblem { inner: problem }),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
}

#[pyclass(name = "Problem")]
pub struct PyProblem {
    inner: chronopt_core::problem::Problem,
}

impl PyProblem {
    pub fn evaluate(&self, x: Vec<f64>) -> f64 {
        self.inner.evaluate(&x)
    }

    pub fn get_config(&self, key: String) -> Option<f64> {
        self.inner.get_config(&key).copied()
    }
}

#[pyclass(name = "NelderMead")]
pub struct PyNelderMead {
    inner: chronopt_core::optimisers::NelderMead,
}

#[pymethods]
impl PyNelderMead {
    #[new]
    fn new() -> Self {
        Self {
            inner: chronopt_core::optimisers::NelderMead::new(),
        }
    }

    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let result = self.inner.run(&problem.inner, initial);
        PyOptimisationResults { inner: result }
    }
}

#[pyclass(name = "OptimisationResults")]
pub struct PyOptimisationResults {
    inner: chronopt_core::optimisers::OptimisationResults,
}

#[pymethods]
impl PyOptimisationResults {
    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x.clone()
    }

    #[getter]
    fn fun(&self) -> f64 {
        self.inner.fun
    }

    #[getter]
    fn nit(&self) -> usize {
        self.inner.nit
    }

    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    // fn __repr__(&self) -> String {
    //     self.inner.to_string()
    // }
}

// Module registration
#[pymodule]
fn chronopt(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBuilder>()?;
    m.add_class::<PyProblem>()?;
    m.add_class::<PyNelderMead>()?;
    m.add_class::<PyOptimisationResults>()?;
    Ok(())
}
