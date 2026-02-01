use nalgebra::DMatrix;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

use diffid_core::builders::{
    DiffsolBackend, DiffsolProblemBuilder, ScalarProblemBuilder, VectorProblemBuilder,
};
use diffid_core::common::Unbounded;
use diffid_core::problem::{NoFunction, NoGradient};

#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::optimisers::Optimiser;
use crate::{DynProblem, PyCostMetric, PyProblem};

// Type aliases
type BoxedScalarFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
type BoxedGradientFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;
type ParameterSpec = (String, f64, Option<(f64, f64)>);
type ScalarBuilderWithFn = ScalarProblemBuilder<BoxedScalarFn, NoGradient>;
type ScalarBuilderWithGrad = ScalarProblemBuilder<BoxedScalarFn, BoxedGradientFn>;

// Python Objective Function Wrapper
pub(crate) struct PyObjectiveFn {
    callable: Py<PyAny>,
}

impl PyObjectiveFn {
    pub fn new(callable: Py<PyAny>) -> Self {
        Self { callable }
    }

    pub fn call(&self, x: &[f64]) -> PyResult<f64> {
        Python::attach(|py| {
            let callable = self.callable.bind(py);
            let input = PyArray1::from_slice(py, x);
            let result = callable.call1((input,))?;

            if let Ok(output) = result.extract::<PyReadonlyArray1<f64>>() {
                let array = output.as_array();
                return match array.len() {
                    1 => Ok(array[0]),
                    n => Err(PyValueError::new_err(format!(
                        "Objective array must contain exactly one element, got {n}"
                    ))),
                };
            }

            if let Ok(values) = result.extract::<Vec<f64>>() {
                return match values.len() {
                    1 => Ok(values[0]),
                    n => Err(PyValueError::new_err(format!(
                        "Objective sequence must contain exactly one element, got {n}"
                    ))),
                };
            }

            if let Ok(value) = result.extract::<f64>() {
                return Ok(value);
            }

            let ty_name = result
                .get_type()
                .name()
                .map_or_else(|_| "unknown".to_string(), |n| n.to_string());

            Err(PyTypeError::new_err(format!(
                "Objective callable must return a float, numpy array, or single-element sequence; got {ty_name}"
            )))
        })
    }
}

pub(crate) struct PyGradientFn {
    callable: Py<PyAny>,
}

impl PyGradientFn {
    pub fn new(callable: Py<PyAny>) -> Self {
        Self { callable }
    }

    pub fn call(&self, x: &[f64]) -> PyResult<Vec<f64>> {
        Python::attach(|py| {
            let callable = self.callable.bind(py);
            let input = PyArray1::from_slice(py, x);
            let result = callable.call1((input,))?;

            if let Ok(output) = result.extract::<PyReadonlyArray1<f64>>() {
                return Ok(output.as_array().to_vec());
            }

            result.extract::<Vec<f64>>()
        })
    }
}

// Builder
// Internal enum to track builder state due to Rust's type-state pattern
enum ScalarBuilderState {
    Empty(ScalarProblemBuilder<NoFunction, NoGradient>),
    WithFunction {
        builder: ScalarBuilderWithFn,
        py_callable: Arc<PyObjectiveFn>,
    },
    WithGradient {
        builder: ScalarBuilderWithGrad,
        py_callable: Arc<PyObjectiveFn>,
        py_gradient: Arc<PyGradientFn>,
    },
}

/// High-level builder for optimisation `Problem` instances exposed to Python.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "ScalarBuilder")]
pub struct PyScalarBuilder {
    state: ScalarBuilderState,
    pub(crate) default_optimiser: Option<Optimiser>,
    parameter_specs: Vec<ParameterSpec>,
    config: HashMap<String, f64>,
}

impl Default for PyScalarBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyScalarBuilder {
    /// Create an empty builder with no objective, parameters, or default optimiser.
    #[new]
    pub fn new() -> Self {
        Self {
            state: ScalarBuilderState::Empty(ScalarProblemBuilder::new()),
            default_optimiser: None,
            parameter_specs: Vec::new(),
            config: HashMap::new(),
        }
    }

    fn __copy__(&self) -> Self {
        Self {
            state: match &self.state {
                ScalarBuilderState::Empty(_) => {
                    ScalarBuilderState::Empty(ScalarProblemBuilder::new())
                }
                ScalarBuilderState::WithFunction { py_callable, .. } => {
                    // Recreate the builder with a new closure from the Arc
                    let objective = Arc::clone(py_callable);
                    let boxed_fn: BoxedScalarFn =
                        Box::new(move |x: &[f64]| objective.call(x).unwrap_or(f64::INFINITY));

                    let mut builder = ScalarProblemBuilder::new().with_function(boxed_fn);

                    // Add parameters
                    for (name, initial, bounds) in &self.parameter_specs {
                        if let Some((l, u)) = bounds {
                            builder = builder.with_parameter(name, *initial, (*l, *u));
                        } else {
                            builder = builder.with_parameter(name, *initial, Unbounded);
                        }
                    }

                    ScalarBuilderState::WithFunction {
                        builder,
                        py_callable: Arc::clone(py_callable),
                    }
                }
                ScalarBuilderState::WithGradient {
                    py_callable,
                    py_gradient,
                    ..
                } => {
                    // Recreate both closures
                    let objective = Arc::clone(py_callable);
                    let boxed_fn: BoxedScalarFn =
                        Box::new(move |x: &[f64]| objective.call(x).unwrap_or(f64::INFINITY));

                    let grad = Arc::clone(py_gradient);
                    let boxed_grad: BoxedGradientFn = Box::new(move |x: &[f64]| {
                        grad.call(x).unwrap_or_else(|_| vec![f64::NAN; x.len()])
                    });

                    let mut builder = ScalarProblemBuilder::new()
                        .with_function(boxed_fn)
                        .with_gradient(boxed_grad);

                    // Add parameters
                    for (name, initial, bounds) in &self.parameter_specs {
                        if let Some((l, u)) = bounds {
                            builder = builder.with_parameter(name, *initial, (*l, *u));
                        } else {
                            builder = builder.with_parameter(name, *initial, Unbounded);
                        }
                    }

                    ScalarBuilderState::WithGradient {
                        builder,
                        py_callable: Arc::clone(py_callable),
                        py_gradient: Arc::clone(py_gradient),
                    }
                }
            },
            default_optimiser: self.default_optimiser.clone(),
            parameter_specs: self.parameter_specs.clone(),
            config: self.config.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.__copy__()
    }

    /// Configure the default optimiser used when `Problem.optimise` omits one.
    fn with_optimiser(mut slf: PyRefMut<'_, Self>, optimiser: Optimiser) -> PyRefMut<'_, Self> {
        let core_opt = optimiser.to_core();
        slf.state = match std::mem::replace(
            &mut slf.state,
            ScalarBuilderState::Empty(ScalarProblemBuilder::new()),
        ) {
            ScalarBuilderState::Empty(builder) => {
                ScalarBuilderState::Empty(builder.with_optimiser(core_opt.clone()))
            }
            ScalarBuilderState::WithFunction {
                builder,
                py_callable,
            } => ScalarBuilderState::WithFunction {
                builder: builder.with_optimiser(core_opt.clone()),
                py_callable,
            },
            ScalarBuilderState::WithGradient {
                builder,
                py_callable,
                py_gradient,
            } => ScalarBuilderState::WithGradient {
                builder: builder.with_optimiser(core_opt.clone()),
                py_callable,
                py_gradient,
            },
        };
        slf.default_optimiser = Some(optimiser);
        slf
    }

    /// Attach the objective function callable executed during optimisation.
    fn with_objective(mut slf: PyRefMut<'_, Self>, obj: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
        Python::attach(|py| {
            if !obj.bind(py).is_callable() {
                return Err(PyTypeError::new_err("Object must be callable"));
            }
            Ok(())
        })?;

        let py_fn = Arc::new(PyObjectiveFn::new(obj));
        let objective = Arc::clone(&py_fn);
        let boxed_fn: BoxedScalarFn =
            Box::new(move |x: &[f64]| objective.call(x).unwrap_or(f64::INFINITY));

        slf.state = match std::mem::replace(
            &mut slf.state,
            ScalarBuilderState::Empty(ScalarProblemBuilder::new()),
        ) {
            ScalarBuilderState::Empty(builder) => ScalarBuilderState::WithFunction {
                builder: builder.with_function(boxed_fn),
                py_callable: py_fn,
            },
            _ => return Err(PyValueError::new_err("Callable already set")),
        };

        Ok(slf)
    }

    /// Attach the gradient callable returning derivatives of the objective.
    fn with_gradient(mut slf: PyRefMut<'_, Self>, obj: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
        Python::attach(|py| {
            if !obj.bind(py).is_callable() {
                return Err(PyTypeError::new_err("Object must be callable"));
            }
            Ok(())
        })?;

        let py_grad = Arc::new(PyGradientFn::new(obj));
        let grad = Arc::clone(&py_grad);
        let boxed_grad: BoxedGradientFn =
            Box::new(move |x: &[f64]| grad.call(x).unwrap_or_else(|_| vec![f64::NAN; x.len()]));

        slf.state = match std::mem::replace(
            &mut slf.state,
            ScalarBuilderState::Empty(ScalarProblemBuilder::new()),
        ) {
            ScalarBuilderState::WithFunction {
                builder,
                py_callable,
            } => ScalarBuilderState::WithGradient {
                builder: builder.with_gradient(boxed_grad),
                py_callable,
                py_gradient: py_grad,
            },
            _ => return Err(PyValueError::new_err("Must set callable before gradient")),
        };

        Ok(slf)
    }

    /// Register a named optimisation variable in the order it appears in vectors.
    #[pyo3(signature = (name, initial_value, bounds=None))]
    fn with_parameter(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        initial_value: f64,
        bounds: Option<(f64, f64)>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        // Validate bounds if provided
        if let Some((lower, upper)) = bounds {
            if lower >= upper {
                return Err(PyValueError::new_err(format!(
                    "Invalid bounds for parameter '{name}': lower bound ({lower}) must be less than upper bound ({upper})"
                )));
            }
            if !initial_value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "Invalid initial value for parameter '{name}': must be finite, got {initial_value}"
                )));
            }
        }

        // Store parameter spec
        slf.parameter_specs
            .push((name.clone(), initial_value, bounds));

        // Convert Option<(f64, f64)> to ParameterRange
        let range: diffid_core::problem::ParameterRange =
            bounds.map_or_else(|| Unbounded.into(), std::convert::Into::into);

        slf.state = match std::mem::replace(
            &mut slf.state,
            ScalarBuilderState::Empty(ScalarProblemBuilder::new()),
        ) {
            ScalarBuilderState::Empty(builder) => {
                ScalarBuilderState::Empty(builder.with_parameter(name, initial_value, range))
            }
            ScalarBuilderState::WithFunction {
                builder,
                py_callable,
            } => ScalarBuilderState::WithFunction {
                builder: builder.with_parameter(name, initial_value, range),
                py_callable,
            },
            ScalarBuilderState::WithGradient {
                builder,
                py_callable,
                py_gradient,
            } => ScalarBuilderState::WithGradient {
                builder: builder.with_parameter(name, initial_value, range),
                py_callable,
                py_gradient,
            },
        };
        Ok(slf)
    }

    /// Finalize the builder into an executable `Problem`.
    fn build(&mut self) -> PyResult<PyProblem> {
        // Recreate builder from Arc references to avoid consuming state
        let dyn_problem = match &self.state {
            ScalarBuilderState::Empty(_) => {
                return Err(PyValueError::new_err("Must set callable before building"));
            }
            ScalarBuilderState::WithFunction { py_callable, .. } => {
                // Recreate a fresh builder from the Arc
                let objective = Arc::clone(py_callable);
                let boxed_fn: BoxedScalarFn =
                    Box::new(move |x: &[f64]| objective.call(x).unwrap_or(f64::INFINITY));

                let mut builder = ScalarProblemBuilder::new().with_function(boxed_fn);

                // Add parameters
                for (name, initial, bounds) in &self.parameter_specs {
                    if let Some((l, u)) = bounds {
                        builder = builder.with_parameter(name, *initial, (*l, *u));
                    } else {
                        builder = builder.with_parameter(name, *initial, Unbounded);
                    }
                }

                let problem = builder.build().map_err(crate::errors::build_error_to_py)?;
                DynProblem::Scalar(problem)
            }
            ScalarBuilderState::WithGradient {
                py_callable,
                py_gradient,
                ..
            } => {
                // Recreate fresh builder with both function and gradient
                let objective = Arc::clone(py_callable);
                let boxed_fn: BoxedScalarFn =
                    Box::new(move |x: &[f64]| objective.call(x).unwrap_or(f64::INFINITY));

                let grad = Arc::clone(py_gradient);
                let boxed_grad: BoxedGradientFn = Box::new(move |x: &[f64]| {
                    grad.call(x).unwrap_or_else(|_| vec![f64::NAN; x.len()])
                });

                let mut builder = ScalarProblemBuilder::new()
                    .with_function(boxed_fn)
                    .with_gradient(boxed_grad);

                // Add parameters
                for (name, initial, bounds) in &self.parameter_specs {
                    if let Some((l, u)) = bounds {
                        builder = builder.with_parameter(name, *initial, (*l, *u));
                    } else {
                        builder = builder.with_parameter(name, *initial, Unbounded);
                    }
                }

                let problem = builder.build().map_err(crate::errors::build_error_to_py)?;
                DynProblem::ScalarWithGradient(problem)
            }
        };

        // self.state remains unchanged for next build()

        Ok(PyProblem {
            inner: dyn_problem,
            default_optimiser: self.default_optimiser.clone(),
            default_sampler: None,
            parameter_specs: self.parameter_specs.clone(),
            config: self.config.clone(),
        })
    }
}

// Diffsol Builder
// Helper function to convert numpy arrays to DMatrix
fn convert_array_to_dmatrix(data: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<DMatrix<f64>> {
    let array = data.as_array();
    let array_2d = array
        .into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| PyValueError::new_err("Data array must be two-dimensional"))?;

    let (nrows, ncols) = array_2d.dim();
    let mut column_major = Vec::with_capacity(nrows * ncols);

    // nalgebra uses column-major storage
    for col in 0..ncols {
        for row in 0..nrows {
            column_major.push(array_2d[[row, col]]);
        }
    }

    Ok(DMatrix::from_vec(nrows, ncols, column_major))
}

/// Differential equation solver builder.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "DiffsolBuilder")]
pub struct PyDiffsolBuilder {
    inner: DiffsolProblemBuilder,
    pub(crate) default_optimiser: Option<Optimiser>,
    parameter_specs: Vec<ParameterSpec>,
    config: HashMap<String, f64>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyDiffsolBuilder {
    /// Create an empty differential solver builder.
    #[new]
    fn new() -> Self {
        Self {
            inner: DiffsolProblemBuilder::new(),
            default_optimiser: None,
            parameter_specs: Vec::new(),
            config: HashMap::new(),
        }
    }

    fn __copy__(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            default_optimiser: self.default_optimiser.clone(),
            parameter_specs: self.parameter_specs.clone(),
            config: self.config.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.__copy__()
    }

    /// Register the `DiffSL` program describing the system dynamics.
    fn with_diffsl(mut slf: PyRefMut<'_, Self>, dsl: String) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_diffsl(dsl);
        slf
    }

    /// Attach observed data used to fit the differential equation.
    ///
    /// The first column must contain the time samples (`t_span`) and the remaining
    /// columns the observed trajectories.
    fn with_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data_matrix = convert_array_to_dmatrix(&data)?;
        if data_matrix.ncols() < 2 {
            return Err(PyValueError::new_err(
                "Data must include at least two columns with t_span in the first column",
            ));
        }
        slf.inner = std::mem::take(&mut slf.inner).with_data(data_matrix);
        Ok(slf)
    }

    /// Remove any previously attached data along with its time span.
    fn remove_data(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).remove_data();
        slf
    }

    /// Choose whether to use dense or sparse diffusion solvers.
    fn with_backend(mut slf: PyRefMut<'_, Self>, backend: String) -> PyResult<PyRefMut<'_, Self>> {
        let backend_enum = match backend.as_str() {
            "dense" => DiffsolBackend::Dense,
            "sparse" => DiffsolBackend::Sparse,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown backend '{other}'. Expected 'dense' or 'sparse'"
                )))
            }
        };
        slf.inner = std::mem::take(&mut slf.inner).with_backend(backend_enum);
        Ok(slf)
    }

    /// Opt into parallel proposal generation when supported by the backend.
    #[pyo3(signature = (parallel=None))]
    fn with_parallel(mut slf: PyRefMut<'_, Self>, parallel: Option<bool>) -> PyRefMut<'_, Self> {
        let parallel = parallel.unwrap_or(true);
        slf.inner = std::mem::take(&mut slf.inner).with_parallel(parallel);
        slf
    }

    fn with_config(
        mut slf: PyRefMut<'_, Self>,
        config: HashMap<String, f64>,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_config(config);
        slf
    }

    /// Adjust the relative and absolute integration tolerances.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, rtol: f64, atol: f64) -> PyRefMut<'_, Self> {
        slf.config.insert("rtol".to_string(), rtol);
        slf.config.insert("atol".to_string(), atol);
        slf.inner = std::mem::take(&mut slf.inner).with_tolerances(rtol, atol);
        slf
    }

    /// Register a named optimisation variable in the order it appears in vectors.
    #[pyo3(signature = (name, initial_value, bounds=None))]
    fn with_parameter(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        initial_value: f64,
        bounds: Option<(f64, f64)>,
    ) -> PyRefMut<'_, Self> {
        slf.parameter_specs
            .push((name.clone(), initial_value, bounds));
        // Note: We don't add to self.inner here - parameters are added during build()
        // from parameter_specs. This allows clear_parameters() to work correctly.
        slf
    }

    /// Clear all previously registered parameters while preserving other configuration.
    fn clear_parameters(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.parameter_specs.clear();
        // Note: We only clear parameter_specs here. The inner builder's parameters
        // are ignored during build() which rebuilds from parameter_specs.
        slf
    }

    /// Select the error metric used to compare simulated and observed data.
    fn with_cost<'py>(
        mut slf: PyRefMut<'py, Self>,
        cost: PyRef<'py, PyCostMetric>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let metric = cost.metric_arc();
        slf.inner = std::mem::take(&mut slf.inner).with_cost_arc(metric);
        Ok(slf)
    }

    /// Reset the cost metric to the default sum of squared errors.
    fn remove_cost(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).remove_costs();
        slf
    }

    /// Configure the default optimiser used when `Problem.optimise` omits one.
    fn with_optimiser(mut slf: PyRefMut<'_, Self>, optimiser: Optimiser) -> PyRefMut<'_, Self> {
        let core_opt = optimiser.to_core();
        slf.inner = std::mem::take(&mut slf.inner).with_optimiser(core_opt);
        slf.default_optimiser = Some(optimiser);
        slf
    }

    /// Create a `Problem` representing the differential solver model.
    fn build(&mut self) -> PyResult<PyProblem> {
        // Clone the builder and add parameters from parameter_specs
        // This ensures clear_parameters() works correctly
        let mut builder = self.inner.clone();

        // Add parameters from parameter_specs (not from the cloned builder)
        for (name, initial, bounds) in &self.parameter_specs {
            if let Some((l, u)) = bounds {
                builder = builder.with_parameter(name, *initial, (*l, *u));
            } else {
                builder = builder.with_parameter(name, *initial, Unbounded);
            }
        }

        let problem = builder.build().map_err(crate::errors::build_error_to_py)?;

        // self.inner remains unchanged for next build()

        Ok(PyProblem {
            inner: DynProblem::Diffsol(problem),
            default_optimiser: self.default_optimiser.clone(),
            default_sampler: None,
            parameter_specs: self.parameter_specs.clone(),
            config: self.config.clone(),
        })
    }
}

// Vector Builder
/// Time-series problem builder for vector-valued objectives.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "VectorBuilder")]
pub struct PyVectorBuilder {
    inner: VectorProblemBuilder,
    pub(crate) default_optimiser: Option<Optimiser>,
    parameter_specs: Vec<ParameterSpec>,
    config: HashMap<String, f64>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyVectorBuilder {
    /// Create an empty vector problem builder.
    #[new]
    fn new() -> Self {
        Self {
            inner: VectorProblemBuilder::new(),
            default_optimiser: None,
            parameter_specs: Vec::new(),
            config: HashMap::new(),
        }
    }

    fn __copy__(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            default_optimiser: self.default_optimiser.clone(),
            parameter_specs: self.parameter_specs.clone(),
            config: self.config.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.__copy__()
    }

    /// Register a callable that produces predictions matching the data shape.
    ///
    /// The callable should accept a parameter vector and return a numpy array
    /// of the same shape as the observed data.
    fn with_objective(
        mut slf: PyRefMut<'_, Self>,
        objective: Py<PyAny>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let obj_fn =
            move |params: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
                Python::attach(|py| {
                    let params_array = PyArray1::from_slice(py, params);
                    let result = objective.call1(py, (params_array,)).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(std::io::Error::other(format!("Objective call failed: {e}")))
                        },
                    )?;

                    let array: PyReadonlyArray1<f64> = result.extract(py).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(std::io::Error::other(format!(
                                "Failed to extract array: {e}"
                            )))
                        },
                    )?;

                    Ok(array
                        .as_slice()
                        .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "Array must be contiguous",
                            ))
                        })?
                        .to_vec())
                })
            };

        slf.inner = std::mem::take(&mut slf.inner).with_function(obj_fn);
        Ok(slf)
    }

    /// Attach observed data used to fit the model.
    ///
    /// The data should be a 1D numpy array. The shape will be inferred
    /// from the data length.
    fn with_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data_vec = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("Data array must be contiguous"))?
            .to_vec();

        slf.inner = std::mem::take(&mut slf.inner).with_data(data_vec);
        Ok(slf)
    }

    /// Register a named optimisation variable in the order it appears in vectors.
    #[pyo3(signature = (name, initial_value, bounds=None))]
    fn with_parameter(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        initial_value: f64,
        bounds: Option<(f64, f64)>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        // Validate bounds if provided
        if let Some((lower, upper)) = bounds {
            if lower >= upper {
                return Err(PyValueError::new_err(format!(
                    "Invalid bounds for parameter '{name}': lower bound ({lower}) must be less than upper bound ({upper})"
                )));
            }
            if !initial_value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "Invalid initial value for parameter '{name}': must be finite, got {initial_value}"
                )));
            }
        }

        slf.parameter_specs
            .push((name.clone(), initial_value, bounds));
        // Note: We don't add to self.inner here - parameters are added during build()
        // from parameter_specs. This allows parameter modifications to work correctly.
        Ok(slf)
    }

    /// Select the error metric used to compare predictions and observed data.
    fn with_cost<'py>(
        mut slf: PyRefMut<'py, Self>,
        cost: PyRef<'py, PyCostMetric>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let metric = cost.metric_arc();
        slf.inner = std::mem::take(&mut slf.inner).with_cost_arc(metric);
        Ok(slf)
    }

    /// Reset the cost metric to the default sum of squared errors.
    fn remove_cost(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).remove_costs();
        slf
    }

    /// Configure the default optimiser used when `Problem.optimise` omits one.
    fn with_optimiser(mut slf: PyRefMut<'_, Self>, optimiser: Optimiser) -> PyRefMut<'_, Self> {
        let mut inner = std::mem::take(&mut slf.inner);
        match &optimiser {
            Optimiser::NelderMead(nm) => {
                inner = inner.with_optimiser(nm.clone());
            }
            Optimiser::Cmaes(cma) => {
                inner = inner.with_optimiser(cma.clone());
            }
            Optimiser::Adam(adam) => {
                inner = inner.with_optimiser(adam.clone());
            }
        }
        slf.inner = inner;
        slf.default_optimiser = Some(optimiser);
        slf
    }

    /// Attach an arbitrary configuration value to the problem.
    fn with_config(mut slf: PyRefMut<'_, Self>, key: String, value: f64) -> PyRefMut<'_, Self> {
        slf.config.insert(key, value);
        slf
    }

    /// Create a `Problem` representing the vector optimisation model.
    fn build(&mut self) -> PyResult<PyProblem> {
        // Clone the builder and add parameters from parameter_specs
        // This ensures parameter modifications work correctly
        let mut builder = self.inner.clone();

        // Add parameters from parameter_specs (not from the cloned builder)
        for (name, initial, bounds) in &self.parameter_specs {
            if let Some((l, u)) = bounds {
                builder = builder.with_parameter(name, *initial, (*l, *u));
            } else {
                builder = builder.with_parameter(name, *initial, Unbounded);
            }
        }

        let problem = builder.build().map_err(crate::errors::build_error_to_py)?;

        // self.inner remains unchanged for next build()

        Ok(PyProblem {
            inner: DynProblem::Vector(problem),
            default_optimiser: self.default_optimiser.clone(),
            default_sampler: None,
            parameter_specs: self.parameter_specs.clone(),
            config: self.config.clone(),
        })
    }
}
