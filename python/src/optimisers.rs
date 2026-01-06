use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::time::Duration;

use chronopt_core::common::{AskResult, Bounds};
use chronopt_core::optimisers::{AdamState, CMAESState, NelderMeadState};
use chronopt_core::prelude::*;

#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
#[cfg(feature = "stubgen")]
use pyo3_stub_gen::TypeInfo;

use crate::errors::tell_error_to_py;
use crate::{PyDone, PyEvaluate, PyOptimisationResults, PyProblem};

// Optimiser Enum for Polymorphic Types
#[cfg(feature = "stubgen")]
pyo3_stub_gen::impl_stub_type!(Optimiser = PyNelderMead | PyCMAES | PyAdam);

#[derive(Clone)]
pub(crate) enum Optimiser {
    NelderMead(NelderMead),
    Cmaes(CMAES),
    Adam(Adam),
}

#[cfg(feature = "stubgen")]
#[allow(dead_code)]
pub(crate) fn optimiser_type_info() -> TypeInfo {
    TypeInfo::unqualified("chronopt._chronopt.NelderMead")
        | TypeInfo::unqualified("chronopt._chronopt.CMAES")
        | TypeInfo::unqualified("chronopt._chronopt.Adam")
}

impl FromPyObject<'_, '_> for Optimiser {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(nm) = obj.extract::<PyRef<PyNelderMead>>() {
            Ok(Optimiser::NelderMead((*nm).inner.clone()))
        } else if let Ok(cma) = obj.extract::<PyRef<PyCMAES>>() {
            Ok(Optimiser::Cmaes((*cma).inner.clone()))
        } else if let Ok(adam) = obj.extract::<PyRef<PyAdam>>() {
            Ok(Optimiser::Adam((*adam).inner.clone()))
        } else {
            Err(PyTypeError::new_err(
                "Optimiser must be an instance of NelderMead, CMAES, or Adam",
            ))
        }
    }
}

impl Optimiser {
    /// Convert to the core Optimiser enum
    pub(crate) fn to_core(&self) -> chronopt_core::optimisers::Optimiser {
        match self {
            Optimiser::NelderMead(nm) => nm.clone().into(),
            Optimiser::Cmaes(cma) => cma.clone().into(),
            Optimiser::Adam(adam) => adam.clone().into(),
        }
    }
}

// NelderMead
/// Classic simplex-based direct search optimiser.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "NelderMead")]
#[derive(Clone)]
pub struct PyNelderMead {
    pub(crate) inner: NelderMead,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyNelderMead {
    /// Create a Nelder-Mead optimiser with default coefficients.
    #[new]
    fn new() -> Self {
        Self {
            inner: NelderMead::new(),
        }
    }

    /// Set the initial global step-size (standard deviation).
    fn with_step_size(mut slf: PyRefMut<'_, Self>, step_size: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_step_size(step_size);
        slf
    }

    /// Limit the number of simplex iterations.
    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    /// Set the stopping threshold on simplex size or objective reduction.
    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    /// Stop once simplex vertices fall within the supplied positional tolerance.
    fn with_position_tolerance(mut slf: PyRefMut<'_, Self>, tolerance: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_position_tolerance(tolerance);
        slf
    }

    /// Abort after evaluating the objective `max_evaluations` times.
    fn with_max_evaluations(
        mut slf: PyRefMut<'_, Self>,
        max_evaluations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_evaluations(max_evaluations);
        slf
    }

    /// Override the reflection, expansion, contraction, and shrink coefficients.
    fn with_coefficients(
        mut slf: PyRefMut<'_, Self>,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_coefficients(alpha, gamma, rho, sigma);
        slf
    }

    /// Abort if the objective fails to improve within the allotted time.
    ///
    /// Parameters
    /// ----------
    /// patience : float or timedelta
    ///     Either seconds (float) or a timedelta object
    #[pyo3(signature = (patience))]
    fn with_patience<'py>(
        mut slf: PyRefMut<'py, Self>,
        patience: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let patience_seconds = if let Ok(duration) = patience.extract::<Duration>() {
            duration.as_secs_f64()
        } else if let Ok(seconds) = patience.extract::<f64>() {
            seconds
        } else {
            return Err(PyTypeError::new_err(
                "patience must be a float (seconds) or timedelta object",
            ));
        };

        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        Ok(slf)
    }

    /// Optimise the given problem starting from the provided initial simplex centre.
    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let opt = Optimiser::NelderMead(self.inner.clone());
        let result = problem.inner.optimise(Some(initial), Some(&opt));
        PyOptimisationResults { inner: result }
    }

    /// Initialize ask-tell optimization state.
    ///
    /// Returns a NelderMeadState object that can be used for incremental optimization
    /// via the ask-tell interface.
    ///
    /// Parameters
    /// ----------
    /// initial : list[float]
    ///     Initial parameter vector (simplex center)
    /// bounds : list[tuple[float, float]], optional
    ///     Parameter bounds as [(lower, upper), ...]. If None, unbounded.
    ///
    /// Returns
    /// -------
    /// NelderMeadState
    ///     State object for ask-tell optimization
    ///
    /// Examples
    /// --------
    /// >>> optimiser = chronopt.NelderMead()
    /// >>> state = optimiser.init(initial=[1.0, 2.0])
    /// >>> while True:
    /// ...     result = state.ask()
    /// ...     if isinstance(result, chronopt.Done):
    /// ...         break
    /// ...     values = [evaluate(pt) for pt in result.points]
    /// ...     state.tell(values)
    #[pyo3(signature = (initial, bounds=None))]
    fn init(
        &self,
        initial: Vec<f64>,
        bounds: Option<Vec<(f64, f64)>>,
    ) -> PyResult<PyNelderMeadState> {
        let bounds = bounds
            .map(Bounds::new)
            .unwrap_or_else(|| Bounds::unbounded_like(&initial));

        let (state, _initial_point) = self.inner.init(initial, bounds);
        Ok(PyNelderMeadState { inner: state })
    }
}

// CMAES Optimiser
/// Covariance Matrix Adaptation Evolution Strategy optimiser.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "CMAES")]
#[derive(Clone)]
pub struct PyCMAES {
    pub(crate) inner: CMAES,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyCMAES {
    /// Create a CMA-ES optimiser with library defaults.
    #[new]
    fn new() -> Self {
        Self {
            inner: CMAES::new(),
        }
    }

    /// Limit the number of iterations/generations before termination.
    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    /// Set the stopping threshold on the best objective value.
    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    /// Set the initial global step-size (standard deviation).
    fn with_step_size(mut slf: PyRefMut<'_, Self>, step_size: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_step_size(step_size);
        slf
    }

    /// Abort the run if no improvement occurs for the given wall-clock duration.
    ///
    /// Parameters
    /// ----------
    /// patience : float or timedelta
    ///     Either seconds (float) or a timedelta object
    #[pyo3(signature = (patience))]
    fn with_patience<'py>(
        mut slf: PyRefMut<'py, Self>,
        patience: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let patience_seconds = if let Ok(duration) = patience.extract::<Duration>() {
            duration.as_secs_f64()
        } else if let Ok(seconds) = patience.extract::<f64>() {
            seconds
        } else {
            return Err(PyTypeError::new_err(
                "patience must be a float (seconds) or timedelta object",
            ));
        };

        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        Ok(slf)
    }

    /// Specify the number of offspring evaluated per generation.
    fn with_population_size(
        mut slf: PyRefMut<'_, Self>,
        population_size: usize,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_population_size(population_size);
        slf
    }

    /// Initialise the internal RNG for reproducible runs.
    fn with_seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_seed(seed);
        slf
    }

    /// Optimise the given problem starting from the provided mean vector.
    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let opt = Optimiser::Cmaes(self.inner.clone());
        let result = problem.inner.optimise(Some(initial), Some(&opt));
        PyOptimisationResults { inner: result }
    }

    /// Initialize ask-tell optimization state.
    ///
    /// Returns a CMAESState object that can be used for incremental optimization
    /// via the ask-tell interface.
    ///
    /// Parameters
    /// ----------
    /// initial : list[float]
    ///     Initial mean vector for the search distribution
    /// bounds : list[tuple[float, float]], optional
    ///     Parameter bounds as [(lower, upper), ...]. If None, unbounded.
    ///
    /// Returns
    /// -------
    /// CMAESState
    ///     State object for ask-tell optimization
    ///
    /// Examples
    /// --------
    /// >>> optimiser = chronopt.CMAES()
    /// >>> state = optimiser.init(initial=[1.0, 2.0])
    /// >>> while True:
    /// ...     result = state.ask()
    /// ...     if isinstance(result, chronopt.Done):
    /// ...         break
    /// ...     values = [evaluate(pt) for pt in result.points]
    /// ...     state.tell(values)
    #[pyo3(signature = (initial, bounds=None))]
    fn init(&self, initial: Vec<f64>, bounds: Option<Vec<(f64, f64)>>) -> PyResult<PyCMAESState> {
        let bounds = bounds
            .map(Bounds::new)
            .unwrap_or_else(|| Bounds::unbounded_like(&initial));

        let (state, _initial_point) = self.inner.init(initial, bounds);
        Ok(PyCMAESState { inner: state })
    }
}

// Adam
/// Adaptive Moment Estimation (Adam) gradient-based optimiser.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "Adam")]
#[derive(Clone)]
pub struct PyAdam {
    pub(crate) inner: Adam,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyAdam {
    /// Create an Adam optimiser with library defaults.
    #[new]
    fn new() -> Self {
        Self { inner: Adam::new() }
    }

    /// Limit the maximum number of optimisation iterations.
    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    /// Set the stopping threshold on the gradient norm.
    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    /// Configure the base learning rate / step size.
    fn with_step_size(mut slf: PyRefMut<'_, Self>, step_size: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_step_size(step_size);
        slf
    }

    /// Override the exponential decay rates for the first and second moments.
    fn with_betas(mut slf: PyRefMut<'_, Self>, beta1: f64, beta2: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_betas(beta1, beta2);
        slf
    }

    /// Override the numerical stability constant added to the denominator.
    fn with_eps(mut slf: PyRefMut<'_, Self>, eps: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_eps(eps);
        slf
    }

    /// Abort the run once the patience window has elapsed.
    ///
    /// Parameters
    /// ----------
    /// patience : float or timedelta
    ///     Either seconds (float) or a timedelta object
    #[pyo3(signature = (patience))]
    fn with_patience<'py>(
        mut slf: PyRefMut<'py, Self>,
        patience: Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let patience_seconds = if let Ok(duration) = patience.extract::<Duration>() {
            duration.as_secs_f64()
        } else if let Ok(seconds) = patience.extract::<f64>() {
            seconds
        } else {
            return Err(PyTypeError::new_err(
                "patience must be a float (seconds) or timedelta object",
            ));
        };

        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        Ok(slf)
    }

    /// Optimise the given problem using Adam starting from the provided point.
    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let opt = Optimiser::Adam(self.inner.clone());
        let result = problem.inner.optimise(Some(initial), Some(&opt));
        PyOptimisationResults { inner: result }
    }

    /// Initialize ask-tell optimization state.
    ///
    /// Returns an AdamState object that can be used for incremental optimization
    /// via the ask-tell interface.
    ///
    /// Parameters
    /// ----------
    /// initial : list[float]
    ///     Initial parameter vector
    /// bounds : list[tuple[float, float]], optional
    ///     Parameter bounds as [(lower, upper), ...]. If None, unbounded.
    ///
    /// Returns
    /// -------
    /// AdamState
    ///     State object for ask-tell optimization
    ///
    /// Examples
    /// --------
    /// >>> optimiser = chronopt.Adam()
    /// >>> state = optimiser.init(initial=[1.0, 2.0])
    /// >>> while True:
    /// ...     result = state.ask()
    /// ...     if isinstance(result, chronopt.Done):
    /// ...         break
    /// ...     values = [evaluate_with_gradient(pt) for pt in result.points]
    /// ...     state.tell(values)
    #[pyo3(signature = (initial, bounds=None))]
    fn init(&self, initial: Vec<f64>, bounds: Option<Vec<(f64, f64)>>) -> PyResult<PyAdamState> {
        let bounds = bounds
            .map(Bounds::new)
            .unwrap_or_else(|| Bounds::unbounded_like(&initial));

        let (state, _initial_point) = self.inner.init(initial, bounds);
        Ok(PyAdamState { inner: state })
    }
}

// Adam State
/// Ask-tell state for incremental Adam optimization.
///
/// This state object allows step-by-step control over the optimization process.
/// Use `ask()` to get points to evaluate, and `tell()` to provide results.
///
/// Examples
/// --------
/// >>> optimiser = chronopt.Adam().with_max_iter(100)
/// >>> state = optimiser.init(initial=[1.0, 2.0])
/// >>> while True:
/// ...     result = state.ask()
/// ...     if isinstance(result, chronopt.Done):
/// ...         print(f"Final result: {result.result}")
/// ...         break
/// ...     # Adam requires gradient information
/// ...     values = [(f(pt), grad_f(pt)) for pt in result.points]
/// ...     state.tell(values)
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "AdamState")]
pub struct PyAdamState {
    inner: AdamState,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyAdamState {
    /// Get the next action: evaluate points or optimization complete.
    ///
    /// Returns
    /// -------
    /// Evaluate | Done
    ///     Either Evaluate(points) requiring function evaluations,
    ///     or Done(result) indicating completion.
    ///
    /// Examples
    /// --------
    /// >>> result = state.ask()
    /// >>> if isinstance(result, chronopt.Evaluate):
    /// ...     print(f"Need to evaluate {len(result.points)} points")
    /// >>> elif isinstance(result, chronopt.Done):
    /// ...     print(f"Optimization complete: {result.result}")
    fn ask(&self, py: Python<'_>) -> Py<PyAny> {
        match self.inner.ask() {
            AskResult::Evaluate(points) => Py::new(py, PyEvaluate { points }).unwrap().into_any(),
            AskResult::Done(results) => Py::new(py, PyDone::with_optimisation_results(py, results))
                .unwrap()
                .into_any(),
        }
    }

    /// Provide evaluation results (value and gradient) for the requested points.
    ///
    /// Parameters
    /// ----------
    /// result : tuple[float, list[float]]
    ///     Tuple of (value, gradient) where gradient is a list of partial derivatives.
    ///     Adam requires gradient information.
    ///
    /// Raises
    /// ------
    /// TellError
    ///     If called after optimization has terminated or if result format is invalid
    /// EvaluationError
    ///     If the evaluation failed or contained invalid values
    ///
    /// Examples
    /// --------
    /// >>> result = state.ask()
    /// >>> if isinstance(result, chronopt.Evaluate):
    /// ...     point = result.points[0]
    /// ...     value = objective(point)
    /// ...     gradient = compute_gradient(point)
    /// ...     state.tell((value, gradient))
    fn tell(&mut self, result: (f64, Vec<f64>)) -> PyResult<()> {
        let (value, gradient) = result;
        self.inner.tell((value, gradient)).map_err(tell_error_to_py)
    }

    /// Get the current iteration count.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of iterations completed
    fn iterations(&self) -> usize {
        self.inner.iterations()
    }

    /// Get the total number of function evaluations.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of function evaluations performed
    fn evaluations(&self) -> usize {
        self.inner.evaluations()
    }

    /// Get the current best point and value found so far.
    ///
    /// Returns
    /// -------
    /// tuple[list[float], float] | None
    ///     (best_point, best_value) or None if no valid evaluations yet
    fn best(&self) -> Option<(Vec<f64>, f64)> {
        self.inner
            .best()
            .map(|(point, value)| (point.to_vec(), value))
    }

    /// Get the current parameter position.
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Current parameter vector
    fn current_position(&self) -> Vec<f64> {
        self.inner.current_position().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "AdamState(iterations={}, evaluations={}, best={})",
            self.inner.iterations(),
            self.inner.evaluations(),
            match self.inner.best() {
                Some((_, value)) => format!("{:.6}", value),
                None => "None".to_string(),
            }
        )
    }

    fn __str__(&self) -> String {
        format!("Adam optimiser at iteration {}", self.inner.iterations())
    }
}

// Nelder-Mead State
/// Ask-tell state for incremental Nelder-Mead optimization.
///
/// This state object allows step-by-step control over the optimization process.
/// Use `ask()` to get points to evaluate, and `tell()` to provide results.
///
/// Examples
/// --------
/// >>> optimiser = chronopt.NelderMead().with_max_iter(100)
/// >>> state = optimiser.init(initial=[1.0, 2.0])
/// >>> while True:
/// ...     result = state.ask()
/// ...     if isinstance(result, chronopt.Done):
/// ...         print(f"Final result: {result.result}")
/// ...         break
/// ...     values = [f(pt) for pt in result.points]
/// ...     state.tell(values)
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "NelderMeadState")]
pub struct PyNelderMeadState {
    inner: NelderMeadState,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyNelderMeadState {
    /// Get the next action: evaluate points or optimization complete.
    ///
    /// Returns
    /// -------
    /// Evaluate | Done
    ///     Either Evaluate(points) requiring function evaluations,
    ///     or Done(result) indicating completion.
    fn ask(&self, py: Python<'_>) -> Py<PyAny> {
        match self.inner.ask() {
            AskResult::Evaluate(points) => Py::new(py, PyEvaluate { points }).unwrap().into_any(),
            AskResult::Done(results) => Py::new(py, PyDone::with_optimisation_results(py, results))
                .unwrap()
                .into_any(),
        }
    }

    /// Provide evaluation result (scalar value) for the requested point.
    ///
    /// Parameters
    /// ----------
    /// result : float
    ///     Scalar objective function value
    ///
    /// Raises
    /// ------
    /// TellError
    ///     If called after optimization has terminated
    /// EvaluationError
    ///     If the evaluation failed or contained invalid values
    fn tell(&mut self, result: f64) -> PyResult<()> {
        self.inner.tell(result).map_err(tell_error_to_py)
    }

    /// Get the current iteration count.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of iterations completed
    fn iterations(&self) -> usize {
        self.inner.iterations()
    }

    /// Get the total number of function evaluations.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of function evaluations performed
    fn evaluations(&self) -> usize {
        self.inner.evaluations()
    }

    /// Get the current best point and value from the simplex.
    ///
    /// Returns
    /// -------
    /// tuple[list[float], float] | None
    ///     (best_point, best_value) or None if no valid evaluations yet
    fn best(&self) -> Option<(Vec<f64>, f64)> {
        self.inner
            .best()
            .map(|(point, value)| (point.to_vec(), value))
    }

    fn __repr__(&self) -> String {
        format!(
            "NelderMeadState(iterations={}, evaluations={}, best={})",
            self.inner.iterations(),
            self.inner.evaluations(),
            match self.inner.best() {
                Some((_, value)) => format!("{:.6}", value),
                None => "None".to_string(),
            }
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Nelder-Mead optimiser at iteration {}",
            self.inner.iterations()
        )
    }
}

// CMAES State
/// Ask-tell state for incremental CMA-ES optimization.
///
/// This state object allows step-by-step control over the optimization process.
/// Use `ask()` to get a population of points to evaluate, and `tell()` to provide results.
///
/// Examples
/// --------
/// >>> optimiser = chronopt.CMAES().with_max_iter(100)
/// >>> state = optimiser.init(initial=[1.0, 2.0])
/// >>> while True:
/// ...     result = state.ask()
/// ...     if isinstance(result, chronopt.Done):
/// ...         print(f"Final result: {result.result}")
/// ...         break
/// ...     values = [f(pt) for pt in result.points]
/// ...     state.tell(values)
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "CMAESState")]
pub struct PyCMAESState {
    inner: CMAESState,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyCMAESState {
    /// Get the next action: evaluate points or optimization complete.
    ///
    /// Returns
    /// -------
    /// Evaluate | Done
    ///     Either Evaluate(points) requiring function evaluations,
    ///     or Done(result) indicating completion.
    ///
    /// Notes
    /// -----
    /// CMA-ES evaluates a population of points each iteration. The number
    /// of points returned depends on the population_size setting.
    fn ask(&self, py: Python<'_>) -> Py<PyAny> {
        match self.inner.ask() {
            AskResult::Evaluate(points) => Py::new(py, PyEvaluate { points }).unwrap().into_any(),
            AskResult::Done(results) => Py::new(py, PyDone::with_optimisation_results(py, results))
                .unwrap()
                .into_any(),
        }
    }

    /// Provide evaluation results for the requested population of points.
    ///
    /// Parameters
    /// ----------
    /// results : list[float]
    ///     List of objective function values corresponding to the points
    ///     from the last ask() call. Must match the number of points.
    ///
    /// Raises
    /// ------
    /// TellError
    ///     If called after optimization has terminated or if wrong number
    ///     of results provided
    /// EvaluationError
    ///     If evaluations failed or contained invalid values
    fn tell(&mut self, results: Vec<f64>) -> PyResult<()> {
        self.inner.tell(results).map_err(tell_error_to_py)
    }

    /// Get the current iteration (generation) count.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of generations completed
    fn iterations(&self) -> usize {
        self.inner.iterations()
    }

    /// Get the total number of function evaluations.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of function evaluations performed
    fn evaluations(&self) -> usize {
        self.inner.evaluations()
    }

    /// Get the current best point and value found so far.
    ///
    /// Returns
    /// -------
    /// tuple[list[float], float] | None
    ///     (best_point, best_value) or None if no valid evaluations yet
    fn best(&self) -> Option<(Vec<f64>, f64)> {
        self.inner
            .best()
            .map(|(point, value)| (point.to_vec(), value))
    }

    /// Get the current mean of the search distribution.
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Current mean vector
    fn mean(&self) -> Vec<f64> {
        self.inner.mean().to_vec()
    }

    /// Get the current step size (sigma).
    ///
    /// Returns
    /// -------
    /// float
    ///     Current global step size
    fn sigma(&self) -> f64 {
        self.inner.sigma()
    }

    fn __repr__(&self) -> String {
        format!(
            "CMAESState(iterations={}, evaluations={}, best={}, sigma={:.6})",
            self.inner.iterations(),
            self.inner.evaluations(),
            match self.inner.best() {
                Some((_, value)) => format!("{:.6}", value),
                None => "None".to_string(),
            },
            self.inner.sigma()
        )
    }

    fn __str__(&self) -> String {
        format!("CMA-ES optimiser at generation {}", self.inner.iterations())
    }
}
