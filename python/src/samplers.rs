use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::time::Duration;

use chronopt_core::common::{AskResult, Bounds};
use chronopt_core::sampler::{
    DynamicNestedSampler as CoreDynamicNestedSampler,
    DynamicNestedSamplerState as CoreDynamicNestedSamplerState,
    MetropolisHastings as CoreMetropolisHastings,
    MetropolisHastingsState as CoreMetropolisHastingsState, NestedSamples as CoreNestedSamples,
    Sampler as CoreSampler, Samples as CoreSamples, SamplingResults,
    ScalarSampler as CoreScalarSampler,
};

#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
#[cfg(feature = "stubgen")]
use pyo3_stub_gen::TypeInfo;

use crate::errors::tell_error_to_py;
use crate::{PyDone, PyEvaluate, PyProblem};

// Sampler Enum for Polymorphic Types
#[cfg(feature = "stubgen")]
pyo3_stub_gen::impl_stub_type!(Sampler = PyMetropolisHastings | PyDynamicNestedSampler);

#[derive(Clone)]
pub(crate) enum Sampler {
    MetropolisHastings(CoreMetropolisHastings),
    DynamicNested(CoreDynamicNestedSampler),
}

#[cfg(feature = "stubgen")]
#[allow(dead_code)]
pub(crate) fn sampler_type_info() -> TypeInfo {
    TypeInfo::unqualified("chronopt._chronopt.MetropolisHastings")
        | TypeInfo::unqualified("chronopt._chronopt.DynamicNestedSampler")
}

impl FromPyObject<'_, '_> for Sampler {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(mh) = obj.extract::<PyRef<PyMetropolisHastings>>() {
            Ok(Sampler::MetropolisHastings((*mh).inner.clone()))
        } else if let Ok(dns) = obj.extract::<PyRef<PyDynamicNestedSampler>>() {
            Ok(Sampler::DynamicNested((*dns).inner.clone()))
        } else {
            Err(PyTypeError::new_err(
                "Sampler must be an instance of MetropolisHastings or DynamicNestedSampler",
            ))
        }
    }
}

impl Sampler {
    /// Convert to the core Sampler enum
    pub(crate) fn to_core(&self) -> CoreSampler {
        match self {
            Sampler::MetropolisHastings(mh) => {
                CoreSampler::Scalar(CoreScalarSampler::MetropolisHastings(mh.clone()))
            }
            Sampler::DynamicNested(dns) => {
                CoreSampler::Scalar(CoreScalarSampler::DynamicNested(dns.clone()))
            }
        }
    }
}

/// Container for sampler draws and diagnostics.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "Samples")]
pub struct PySamples {
    pub(crate) inner: CoreSamples,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PySamples {
    #[getter]
    fn chains<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        use numpy::ndarray::Array3;

        let rust_chains = self.inner.chains();

        if rust_chains.is_empty() {
            let empty_array = Array3::<f64>::zeros((0, 0, 0));
            return empty_array.to_pyarray(py);
        }

        let n_chains = rust_chains.len();
        let n_iterations = rust_chains[0].len();
        let n_params = rust_chains[0][0].len();

        // Create ndarray from nested vec
        let mut array = Array3::<f64>::zeros((n_chains, n_iterations, n_params));
        for (i, chain) in rust_chains.iter().enumerate() {
            for (j, sample) in chain.iter().enumerate() {
                for (k, &value) in sample.iter().enumerate() {
                    array[[i, j, k]] = value;
                }
            }
        }

        array.to_pyarray(py)
    }

    #[getter]
    fn samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_vec2(py, &self.inner.samples()).expect("Valid Array2")
    }

    #[getter]
    fn mean_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.mean_x().to_pyarray(py)
    }

    #[getter]
    fn acceptance_rate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let rates = self.inner.acceptance_rates();
        rates.to_pyarray(py)
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.draws()
    }

    #[getter]
    fn time(&self) -> Duration {
        self.inner.time()
    }

    fn __repr__(&self) -> String {
        format!(
            "Samples(draws={}, mean_x={:?}, chains={}, time={:?})",
            self.inner.draws(),
            self.inner.mean_x(),
            self.inner.chains().len(),
            self.inner.time()
        )
    }

    /// Return a human-readable summary of the samples.
    fn __str__(&self) -> String {
        format!(
            "{} chains with {} draws each",
            self.inner.chains().len(),
            self.inner.draws()
        )
    }

    /// Return the number of chains.
    fn __len__(&self) -> usize {
        self.inner.chains().len()
    }

    /// Iterate over chains.
    ///
    /// Yields each chain as a list of samples.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<SamplesIterator> {
        Ok(SamplesIterator {
            chains: slf.inner.chains().to_vec(),
            index: 0,
        })
    }

    /// Get a specific chain by index.
    ///
    /// Parameters
    /// ----------
    /// idx : int
    ///     Chain index (0 to num_chains - 1)
    ///
    /// Returns
    /// -------
    /// list[list[float]]
    ///     The requested chain
    fn __getitem__(&self, idx: isize) -> PyResult<Vec<Vec<f64>>> {
        let chains = self.inner.chains();
        let len = chains.len() as isize;

        let idx = if idx < 0 {
            (len + idx) as usize
        } else {
            idx as usize
        };

        chains.get(idx).cloned().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "chain index out of range: {} (have {} chains)",
                idx,
                chains.len()
            ))
        })
    }
}

/// Iterator for Samples chains
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass]
struct SamplesIterator {
    chains: Vec<Vec<Vec<f64>>>,
    index: usize,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl SamplesIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Vec<Vec<f64>>> {
        if slf.index < slf.chains.len() {
            let chain = slf.chains[slf.index].clone();
            slf.index += 1;
            Some(chain)
        } else {
            None
        }
    }
}

/// Nested sampling results including evidence estimates.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "NestedSamples")]
#[derive(Clone)]
pub struct PyNestedSamples {
    pub(crate) inner: CoreNestedSamples,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyNestedSamples {
    #[getter]
    fn posterior(&self) -> Vec<(Vec<f64>, f64, f64)> {
        self.inner
            .posterior()
            .iter()
            .map(|sample| {
                (
                    sample.position.clone(),
                    sample.log_likelihood,
                    sample.log_weight,
                )
            })
            .collect()
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.mean().to_pyarray(py)
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.draws()
    }

    #[getter]
    fn log_evidence(&self) -> f64 {
        self.inner.log_evidence()
    }

    #[getter]
    fn information(&self) -> f64 {
        self.inner.information()
    }

    #[getter]
    fn time(&self) -> Duration {
        self.inner.time()
    }

    fn to_samples(&self) -> PySamples {
        PySamples {
            inner: self.inner.to_samples(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NestedSamples(draws={}, log_evidence={:.3}, information={:.3})",
            self.inner.draws(),
            self.inner.log_evidence(),
            self.inner.information()
        )
    }

    /// Return a human-readable summary of the nested samples.
    fn __str__(&self) -> String {
        format!(
            "{} samples, log(Z) = {:.3}",
            self.inner.draws(),
            self.inner.log_evidence()
        )
    }

    /// Return the number of posterior samples.
    fn __len__(&self) -> usize {
        self.inner.draws()
    }

    /// Iterate over posterior samples.
    ///
    /// Yields tuples of (position, log_likelihood, log_weight).
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<NestedSamplesIterator> {
        let posterior: Vec<_> = slf
            .inner
            .posterior()
            .iter()
            .map(|sample| {
                (
                    sample.position.clone(),
                    sample.log_likelihood,
                    sample.log_weight,
                )
            })
            .collect();

        Ok(NestedSamplesIterator {
            samples: posterior,
            index: 0,
        })
    }

    /// Get a specific posterior sample by index.
    ///
    /// Parameters
    /// ----------
    /// idx : int
    ///     Sample index (0 to num_samples - 1)
    ///
    /// Returns
    /// -------
    /// tuple[list[float], float, float]
    ///     Tuple of (position, log_likelihood, log_weight)
    fn __getitem__(&self, idx: isize) -> PyResult<(Vec<f64>, f64, f64)> {
        let posterior = self.inner.posterior();
        let len = posterior.len() as isize;

        let idx = if idx < 0 {
            (len + idx) as usize
        } else {
            idx as usize
        };

        posterior
            .get(idx)
            .map(|sample| {
                (
                    sample.position.clone(),
                    sample.log_likelihood,
                    sample.log_weight,
                )
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "sample index out of range: {} (have {} samples)",
                    idx,
                    posterior.len()
                ))
            })
    }
}

/// Iterator for NestedSamples posterior
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass]
struct NestedSamplesIterator {
    samples: Vec<(Vec<f64>, f64, f64)>,
    index: usize,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl NestedSamplesIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(Vec<f64>, f64, f64)> {
        if slf.index < slf.samples.len() {
            let sample = slf.samples[slf.index].clone();
            slf.index += 1;
            Some(sample)
        } else {
            None
        }
    }
}

/// Basic Metropolis-Hastings sampler binding mirroring the optimiser API.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "MetropolisHastings")]
#[derive(Clone)]
pub struct PyMetropolisHastings {
    pub(crate) inner: CoreMetropolisHastings,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyMetropolisHastings {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreMetropolisHastings::new(),
        }
    }

    fn with_num_chains(mut slf: PyRefMut<'_, Self>, num_chains: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_num_chains(num_chains);
        slf
    }

    fn with_iterations(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_iterations(iterations);
        slf
    }

    fn with_step_size(mut slf: PyRefMut<'_, Self>, step_size: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_step_size(step_size);
        slf
    }

    fn with_seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_seed(seed);
        slf
    }

    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PySamples {
        let bounds = problem.inner.bounds();
        let samples = self
            .inner
            .run(|x| problem.inner.evaluate(x), initial, bounds.clone());
        PySamples { inner: samples }
    }

    /// Initialize ask-tell sampling state.
    ///
    /// Returns a MetropolisHastingsState object that can be used for incremental
    /// sampling via the ask-tell interface.
    ///
    /// Parameters
    /// ----------
    /// initial : list[float]
    ///     Initial point for all chains
    /// bounds : list[tuple[float, float]], optional
    ///     Parameter bounds as [(lower, upper), ...]. If None, unbounded.
    ///
    /// Returns
    /// -------
    /// MetropolisHastingsState
    ///     State object for ask-tell sampling
    ///
    /// Examples
    /// --------
    /// >>> sampler = chronopt.MetropolisHastings().with_num_chains(4)
    /// >>> state = sampler.init(initial=[1.0, 2.0])
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
    ) -> PyResult<PyMetropolisHastingsState> {
        let bounds = bounds
            .map(Bounds::new)
            .unwrap_or_else(|| Bounds::unbounded_like(&initial));

        let state = self.inner.init(initial, bounds);
        Ok(PyMetropolisHastingsState { inner: state })
    }
}

/// Dynamic nested sampler binding exposing DNS configuration knobs.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "DynamicNestedSampler")]
#[derive(Clone)]
pub struct PyDynamicNestedSampler {
    pub(crate) inner: CoreDynamicNestedSampler,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyDynamicNestedSampler {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreDynamicNestedSampler::new(),
        }
    }

    fn with_live_points(mut slf: PyRefMut<'_, Self>, live_points: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_live_points(live_points);
        slf
    }

    fn with_expansion_factor(
        mut slf: PyRefMut<'_, Self>,
        expansion_factor: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_expansion_factor(expansion_factor);
        slf
    }

    fn with_termination_tolerance(
        mut slf: PyRefMut<'_, Self>,
        tolerance: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_termination_tolerance(tolerance);
        slf
    }

    fn with_seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_seed(seed);
        slf
    }

    #[pyo3(signature = (problem, initial=None))]
    fn run(&self, problem: &PyProblem, initial: Option<Vec<f64>>) -> PyNestedSamples {
        let initial = initial.unwrap_or_else(|| problem.inner.default_parameters());
        let bounds = problem.inner.bounds();
        let nested = self
            .inner
            .run(|x| problem.inner.evaluate(x), initial, bounds.clone());
        PyNestedSamples { inner: nested }
    }

    /// Initialize ask-tell sampling state.
    ///
    /// Returns a DynamicNestedSamplerState object that can be used for incremental
    /// sampling via the ask-tell interface.
    ///
    /// Parameters
    /// ----------
    /// initial : list[float]
    ///     Initial point for the sampler
    /// bounds : list[tuple[float, float]], optional
    ///     Parameter bounds as [(lower, upper), ...]. If None, unbounded.
    ///
    /// Returns
    /// -------
    /// DynamicNestedSamplerState
    ///     State object for ask-tell sampling
    ///
    /// Examples
    /// --------
    /// >>> sampler = chronopt.DynamicNestedSampler()
    /// >>> state = sampler.init(initial=[1.0, 2.0])
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
    ) -> PyResult<PyDynamicNestedSamplerState> {
        let bounds = bounds
            .map(Bounds::new)
            .unwrap_or_else(|| Bounds::unbounded_like(&initial));

        let (state, _initial_points) = self.inner.init(initial, bounds);
        Ok(PyDynamicNestedSamplerState { inner: state })
    }
}

/// Ask-tell state for incremental Metropolis-Hastings MCMC sampling.
///
/// This state object allows step-by-step control over the sampling process.
/// Use `ask()` to get proposal points to evaluate, and `tell()` to provide results.
///
/// Examples
/// --------
/// >>> sampler = chronopt.MetropolisHastings().with_num_chains(4)
/// >>> state = sampler.init(initial=[1.0, 2.0])
/// >>> while True:
/// ...     result = state.ask()
/// ...     if isinstance(result, chronopt.Done):
/// ...         print(f"Sampling complete: {result.result}")
/// ...         break
/// ...     values = [negative_log_likelihood(pt) for pt in result.points]
/// ...     state.tell(values)
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "MetropolisHastingsState")]
pub struct PyMetropolisHastingsState {
    inner: CoreMetropolisHastingsState,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyMetropolisHastingsState {
    /// Get the next action: evaluate proposal points or sampling complete.
    ///
    /// Returns
    /// -------
    /// Evaluate | Done
    ///     Either Evaluate(points) requiring function evaluations,
    ///     or Done(result) indicating completion with Samples.
    ///
    /// Notes
    /// -----
    /// Returns one proposal point per chain.
    fn ask(&self, py: Python<'_>) -> Py<PyAny> {
        match self.inner.ask() {
            AskResult::Evaluate(points) => Py::new(py, PyEvaluate { points }).unwrap().into_any(),
            AskResult::Done(SamplingResults::MCMC(samples)) => {
                Py::new(py, PyDone::with_samples(py, PySamples { inner: samples }))
                    .unwrap()
                    .into_any()
            }
            _ => unreachable!("MetropolisHastings always returns MCMC results"),
        }
    }

    /// Provide evaluation results for the proposed points.
    ///
    /// Parameters
    /// ----------
    /// results : list[float]
    ///     Negative log-likelihood values for each proposal point.
    ///     Must match the number of chains.
    ///
    /// Raises
    /// ------
    /// TellError
    ///     If called after sampling has terminated or if wrong number
    ///     of results provided
    fn tell(&mut self, results: Vec<f64>) -> PyResult<()> {
        self.inner.tell(results).map_err(tell_error_to_py)
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

    /// Get the number of chains being run.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of parallel MCMC chains
    fn num_chains(&self) -> usize {
        self.inner.num_chains()
    }

    fn __repr__(&self) -> String {
        format!(
            "MetropolisHastingsState(iterations={}, chains={})",
            self.inner.iterations(),
            self.inner.num_chains()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Metropolis-Hastings sampler at iteration {} ({} chains)",
            self.inner.iterations(),
            self.inner.num_chains()
        )
    }
}

/// Ask-tell state for incremental Dynamic Nested Sampling.
///
/// This state object allows step-by-step control over the sampling process.
/// Use `ask()` to get points to evaluate, and `tell()` to provide results.
///
/// Examples
/// --------
/// >>> sampler = chronopt.DynamicNestedSampler()
/// >>> state = sampler.init(initial=[1.0, 2.0])
/// >>> while True:
/// ...     result = state.ask()
/// ...     if isinstance(result, chronopt.Done):
/// ...         print(f"Sampling complete: {result.result}")
/// ...         break
/// ...     values = [negative_log_likelihood(pt) for pt in result.points]
/// ...     state.tell(values)
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "DynamicNestedSamplerState")]
pub struct PyDynamicNestedSamplerState {
    inner: CoreDynamicNestedSamplerState,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyDynamicNestedSamplerState {
    /// Get the next action: evaluate points or sampling complete.
    ///
    /// Returns
    /// -------
    /// Evaluate | Done
    ///     Either Evaluate(points) requiring function evaluations,
    ///     or Done(result) indicating completion with NestedSamples.
    fn ask(&self, py: Python<'_>) -> Py<PyAny> {
        match self.inner.ask() {
            AskResult::Evaluate(points) => Py::new(py, PyEvaluate { points }).unwrap().into_any(),
            AskResult::Done(SamplingResults::Nested(samples)) => Py::new(
                py,
                PyDone::with_nested_samples(py, PyNestedSamples { inner: samples }),
            )
            .unwrap()
            .into_any(),
            _ => unreachable!("DynamicNestedSampler always returns Nested results"),
        }
    }

    /// Provide evaluation results for the requested points.
    ///
    /// Parameters
    /// ----------
    /// results : list[float]
    ///     Negative log-likelihood values for each requested point.
    ///
    /// Raises
    /// ------
    /// TellError
    ///     If called after sampling has terminated or if wrong number
    ///     of results provided
    fn tell(&mut self, results: Vec<f64>) -> PyResult<()> {
        self.inner.tell(results).map_err(tell_error_to_py)
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

    /// Get the number of live points.
    ///
    /// Returns
    /// -------
    /// int
    ///     Current number of live points in the sampler
    fn num_live_points(&self) -> usize {
        self.inner.live_point_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicNestedSamplerState(iterations={}, live_points={})",
            self.inner.iterations(),
            self.inner.live_point_count()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Dynamic nested sampler at iteration {}",
            self.inner.iterations()
        )
    }
}
