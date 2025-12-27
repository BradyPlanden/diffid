use crate::common::{Bounds, Point};
use std::time::Duration;

mod dynamic_nested;
mod errors;
mod metropolis_hastings;

use crate::errors::EvaluationError;
use crate::optimisers::{GradientEvaluation, ScalarEvaluation};
pub use dynamic_nested::{DynamicNestedSampler, NestedSample, NestedSamples};
pub use metropolis_hastings::{MetropolisHastings, MetropolisHastingsState};

/// Samplers that only require objective function values (no gradients)
#[derive(Clone, Debug)]
pub enum ScalarSampler {
    MetropolisHastings(MetropolisHastings),
    DynamicNested(DynamicNestedSampler),
}

/// Samplers that require gradient information (e.g., HMC, NUTS)
///
/// Currently uninhabited - will be populated when gradient-aware samplers are added.
#[derive(Clone, Debug)]
pub enum GradientSampler {
    // Future variants:
    // HMC(HamiltonianMC),
    // NUTS(NoUTurnSampler),
}

/// Primary sampler type supporting both scalar and gradient-based sampling
#[derive(Clone, Debug)]
pub enum Sampler {
    /// Scalar sampler (only requires objective values)
    Scalar(ScalarSampler),
    /// Gradient sampler (requires gradient information)
    Gradient(GradientSampler),
}

impl ScalarSampler {
    /// Run the sampler with a scalar objective function
    ///
    /// # Arguments
    /// * `problem` - Problem defining objective and parameter specs
    /// * `initial` - Initial point for sampling
    /// * `bounds` - Parameter bounds
    ///
    /// # Returns
    /// Sampling results (type depends on sampler algorithm)
    pub fn run<F, R, E>(&self, objective: F, initial: Point, bounds: Bounds) -> SamplingResults
    where
        F: FnMut(&[f64]) -> R,
        R: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        match self {
            ScalarSampler::MetropolisHastings(mh) => {
                // Delegates to individual sampler's run() method
                SamplingResults::MCMC(mh.run(objective, initial, bounds))
            }
            ScalarSampler::DynamicNested(dns) => {
                // Delegates to individual sampler's run() method
                SamplingResults::Nested(dns.run(objective, initial, bounds))
            }
        }
    }
}

impl GradientSampler {
    /// Run the sampler with gradient information
    ///
    /// Note: Currently no gradient samplers implemented.
    /// This will be used for HMC, NUTS, etc.
    pub fn run<F, E, R>(&self, objective: F, initial: Point, bounds: Bounds) -> SamplingResults
    where
        F: FnMut(&[f64]) -> R,
        R: TryInto<GradientEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        // Empty match - uninhabited enum, will be exhaustive when variants added
        match *self {}
    }
}

impl Sampler {
    // ─── Convenience constructors ────────────────────────────────────────────

    /// Create a Metropolis-Hastings sampler with default settings
    pub fn metropolis_hastings() -> Self {
        Sampler::Scalar(ScalarSampler::MetropolisHastings(
            MetropolisHastings::default(),
        ))
    }

    /// Create a Dynamic Nested Sampler with default settings
    pub fn dynamic_nested() -> Self {
        Sampler::Scalar(ScalarSampler::DynamicNested(DynamicNestedSampler::default()))
    }

    // ─── Type extraction - borrowed ──────────────────────────────────────────

    /// Try to get reference to inner scalar sampler
    pub fn as_scalar(&self) -> Option<&ScalarSampler> {
        match self {
            Sampler::Scalar(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get mutable reference to inner scalar sampler
    pub fn as_scalar_mut(&mut self) -> Option<&mut ScalarSampler> {
        match self {
            Sampler::Scalar(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get reference to inner gradient sampler
    pub fn as_gradient(&self) -> Option<&GradientSampler> {
        match self {
            Sampler::Gradient(g) => Some(g),
            _ => None,
        }
    }

    /// Try to get mutable reference to inner gradient sampler
    pub fn as_gradient_mut(&mut self) -> Option<&mut GradientSampler> {
        match self {
            Sampler::Gradient(g) => Some(g),
            _ => None,
        }
    }

    // ─── Type extraction - consuming ─────────────────────────────────────────

    /// Try to convert into a scalar sampler
    pub fn into_scalar(self) -> Result<ScalarSampler, Self> {
        match self {
            Sampler::Scalar(s) => Ok(s),
            other => Err(other),
        }
    }

    /// Try to convert into a gradient sampler
    pub fn into_gradient(self) -> Result<GradientSampler, Self> {
        match self {
            Sampler::Gradient(g) => Ok(g),
            other => Err(other),
        }
    }

    // ─── Type queries ────────────────────────────────────────────────────────

    /// Check if this is a scalar sampler
    pub fn is_scalar(&self) -> bool {
        matches!(self, Sampler::Scalar(_))
    }

    /// Check if this is a gradient sampler
    pub fn is_gradient(&self) -> bool {
        matches!(self, Sampler::Gradient(_))
    }

    /// Get a readable name for the sampler
    pub fn name(&self) -> &'static str {
        match self {
            Sampler::Scalar(ScalarSampler::MetropolisHastings(_)) => "Metropolis-Hastings",
            Sampler::Scalar(ScalarSampler::DynamicNested(_)) => "Dynamic Nested Sampler",
            Sampler::Gradient(_) => {
                // Will need updating when gradient samplers are added
                "Gradient Sampler"
            }
        }
    }
}

// From Trait Implementations
// ─── Individual -> ScalarSampler ──────────────────────────────────────────

impl From<MetropolisHastings> for ScalarSampler {
    fn from(mh: MetropolisHastings) -> Self {
        ScalarSampler::MetropolisHastings(mh)
    }
}

impl From<DynamicNestedSampler> for ScalarSampler {
    fn from(dns: DynamicNestedSampler) -> Self {
        ScalarSampler::DynamicNested(dns)
    }
}

// ─── Sub-enums -> Sampler ─────────────────────────────────────────────────

impl From<ScalarSampler> for Sampler {
    fn from(s: ScalarSampler) -> Self {
        Sampler::Scalar(s)
    }
}

impl From<GradientSampler> for Sampler {
    fn from(g: GradientSampler) -> Self {
        Sampler::Gradient(g)
    }
}

// ─── Individual -> Sampler (convenience) ──────────────────────────────────

impl From<MetropolisHastings> for Sampler {
    fn from(mh: MetropolisHastings) -> Self {
        Sampler::Scalar(ScalarSampler::MetropolisHastings(mh))
    }
}

impl From<DynamicNestedSampler> for Sampler {
    fn from(dns: DynamicNestedSampler) -> Self {
        Sampler::Scalar(ScalarSampler::DynamicNested(dns))
    }
}

impl Default for ScalarSampler {
    fn default() -> Self {
        ScalarSampler::MetropolisHastings(MetropolisHastings::default())
    }
}

// Note: No Default for GradientSampler until variants are added

impl Default for Sampler {
    fn default() -> Self {
        Sampler::Scalar(ScalarSampler::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Result Types (Samples, etc.)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct Samples {
    chains: Vec<Vec<Vec<f64>>>,
    mean_x: Vec<f64>,
    draws: usize,
    time: Duration,
}

impl Samples {
    pub fn new(chains: Vec<Vec<Vec<f64>>>, mean_x: Vec<f64>, draws: usize, time: Duration) -> Self {
        Self {
            chains,
            mean_x,
            draws,
            time,
        }
    }

    pub fn chains(&self) -> &[Vec<Vec<f64>>] {
        &self.chains
    }

    pub fn mean_x(&self) -> &[f64] {
        &self.mean_x
    }

    pub fn draws(&self) -> usize {
        self.draws
    }

    pub fn time(&self) -> Duration {
        self.time
    }
}

/// Unified result type for all samplers
#[derive(Clone, Debug)]
pub enum SamplingResults {
    /// MCMC sampling results (e.g., Metropolis-Hastings)
    MCMC(Samples),
    /// Nested sampling results (e.g., Dynamic Nested Sampler)
    Nested(NestedSamples),
}

impl SamplingResults {
    /// Get number of samples/draws across all result types
    pub fn draws(&self) -> usize {
        match self {
            SamplingResults::MCMC(s) => s.draws(),
            SamplingResults::Nested(s) => s.draws(),
        }
    }

    /// Get execution time across all result types
    pub fn time(&self) -> Duration {
        match self {
            SamplingResults::MCMC(s) => s.time(),
            SamplingResults::Nested(s) => s.time(),
        }
    }

    /// Get posterior mean across all result types
    pub fn mean(&self) -> &[f64] {
        match self {
            SamplingResults::MCMC(s) => s.mean_x(),
            SamplingResults::Nested(s) => s.mean(),
        }
    }

    /// Try to extract MCMC samples
    pub fn as_mcmc(&self) -> Option<&Samples> {
        match self {
            SamplingResults::MCMC(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract MCMC samples (mutable)
    pub fn as_mcmc_mut(&mut self) -> Option<&mut Samples> {
        match self {
            SamplingResults::MCMC(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract nested samples
    pub fn as_nested(&self) -> Option<&NestedSamples> {
        match self {
            SamplingResults::Nested(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract nested samples (mutable)
    pub fn as_nested_mut(&mut self) -> Option<&mut NestedSamples> {
        match self {
            SamplingResults::Nested(s) => Some(s),
            _ => None,
        }
    }

    /// Check if this is MCMC results
    pub fn is_mcmc(&self) -> bool {
        matches!(self, SamplingResults::MCMC(_))
    }

    /// Check if this is nested sampling results
    pub fn is_nested(&self) -> bool {
        matches!(self, SamplingResults::Nested(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::ScalarProblemBuilder;
    use crate::common::{AskResult, Unbounded};
    use crate::errors::TellError;

    #[test]
    fn metropolis_hastings_produces_samples() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| {
                let diff = x[0] - 1.0;
                0.5 * diff * diff
            })
            .with_parameter("x", 1.0, Unbounded)
            .build()
            .expect("problem to build");

        let sampler = MetropolisHastings::new()
            .with_num_chains(4)
            .with_iterations(600)
            .with_step_size(0.3)
            .with_seed(42);

        let samples = sampler.run(|x| problem.evaluate(x), vec![0.0]);

        assert_eq!(samples.chains().len(), 4);
        for chain in samples.chains() {
            assert_eq!(chain.len(), 601);
        }

        let mean = samples.mean_x();
        assert_eq!(mean.len(), 1);
        assert!((mean[0] - 1.0).abs() < 0.2);
        assert_eq!(samples.draws(), 4 * 600);
    }

    #[test]
    fn sampler_enum_scalar_works() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| 0.5 * x[0].powi(2))
            .with_parameter("x", 1.0, Unbounded)
            .build()
            .expect("problem to build");

        let sampler = Sampler::metropolis_hastings();
        let scalar_sampler = sampler.as_scalar().expect("Should be scalar sampler");
        let results = scalar_sampler.run(|x| problem.evaluate(x), vec![0.0]);

        assert!(matches!(results, SamplingResults::MCMC(_)));
        assert!(results.is_mcmc());
        assert!(!results.is_nested());
    }

    #[test]
    fn sampler_type_queries_work() {
        let mh = Sampler::metropolis_hastings();
        assert!(mh.is_scalar());
        assert!(!mh.is_gradient());
        assert_eq!(mh.name(), "Metropolis-Hastings");

        let dns = Sampler::dynamic_nested();
        assert!(dns.is_scalar());
        assert!(!dns.is_gradient());
        assert_eq!(dns.name(), "Dynamic Nested Sampler");
    }

    #[test]
    fn sampler_conversions_work() {
        let mh = MetropolisHastings::new();
        let _: ScalarSampler = mh.clone().into();
        let _: Sampler = mh.into();

        let dns = DynamicNestedSampler::new();
        let _: ScalarSampler = dns.clone().into();
        let _: Sampler = dns.into();
    }

    #[test]
    fn sampling_results_common_methods() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| 0.5 * x[0].powi(2))
            .with_parameter("x", 1.0, Unbounded)
            .build()
            .expect("problem to build");

        let mh = MetropolisHastings::new().with_iterations(100).with_seed(42);
        let results = ScalarSampler::from(mh).run(|x| problem.evaluate(x), vec![0.5]);

        // Common methods work across result types
        assert!(results.draws() > 0);
        assert!(results.time().as_nanos() > 0);
        assert_eq!(results.mean().len(), 1);

        // Type-specific access
        let mcmc_samples = results.as_mcmc().expect("Should be MCMC results");
        assert!(mcmc_samples.chains().len() > 0);
    }

    #[test]
    fn sampling_results_type_checking() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| 0.5 * x[0].powi(2))
            .with_parameter("x", 0.0, (-5.0, 5.0))
            .build()
            .expect("problem to build");

        // MCMC results
        let mh = MetropolisHastings::new().with_seed(42);
        let mcmc_results = ScalarSampler::from(mh).run(|x| problem.evaluate(x), vec![0.0]);
        assert!(mcmc_results.is_mcmc());
        assert!(!mcmc_results.is_nested());
        assert!(mcmc_results.as_mcmc().is_some());
        assert!(mcmc_results.as_nested().is_none());

        // Nested results
        let dns = DynamicNestedSampler::new()
            .with_live_points(32)
            .with_seed(42);
        let nested_results = ScalarSampler::from(dns).run(|x| problem.evaluate(x), vec![0.0]);
        assert!(!nested_results.is_mcmc());
        assert!(nested_results.is_nested());
        assert!(nested_results.as_mcmc().is_none());
        assert!(nested_results.as_nested().is_some());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Ask/Tell Interface Tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn metropolis_hastings_ask_tell_works() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| 0.5 * (x[0] - 1.0).powi(2))
            .with_parameter("x", 1.0, Unbounded)
            .build()
            .unwrap();

        let sampler = MetropolisHastings::new()
            .with_num_chains(2)
            .with_iterations(100)
            .with_seed(42);

        let mut state = sampler.init(vec![0.0]);
        let mut eval_count = 0;

        loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    eval_count += points.len();
                    let results: Vec<_> = points.iter().map(|x| problem.evaluate(x)).collect();
                    state.tell(results).unwrap();
                }
                AskResult::Done(SamplingResults::MCMC(samples)) => {
                    assert_eq!(samples.chains().len(), 2);
                    assert!(eval_count > 0);
                    // Each chain should have initial + iterations samples
                    for chain in samples.chains() {
                        assert_eq!(chain.len(), 101);
                    }
                    break;
                }
                _ => panic!("Expected MCMC results"),
            }
        }
    }

    #[test]
    fn ask_tell_matches_run_results() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| 0.5 * (x[0] - 1.0).powi(2))
            .with_parameter("x", 1.0, Unbounded)
            .build()
            .unwrap();

        let sampler = MetropolisHastings::new()
            .with_num_chains(2)
            .with_iterations(50)
            .with_seed(42);

        // Run using ask/tell
        let mut state = sampler.init(vec![0.0]);
        let ask_tell_results = loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    let results: Vec<_> = points.iter().map(|x| problem.evaluate(x)).collect();
                    state.tell(results).unwrap();
                }
                AskResult::Done(SamplingResults::MCMC(samples)) => break samples,
                _ => panic!("Expected MCMC results"),
            }
        };

        // Run using direct run() method (which now uses ask/tell internally)
        let run_results = sampler.run(|x| problem.evaluate(x), vec![0.0]);

        // Both should produce same structure
        assert_eq!(ask_tell_results.chains().len(), run_results.chains().len());
        assert_eq!(ask_tell_results.draws(), run_results.draws());
        assert_eq!(ask_tell_results.mean_x().len(), run_results.mean_x().len());
    }

    #[test]
    fn ask_tell_error_handling() {
        let sampler = MetropolisHastings::new()
            .with_num_chains(2)
            .with_iterations(1);

        let mut state = sampler.init(vec![0.0]);

        match state.ask() {
            AskResult::Evaluate(_) => {
                // Wrong number of results
                let results: Vec<Result<f64, std::io::Error>> = vec![Ok(1.0)]; // Should be 2
                let err = state.tell(results).unwrap_err();
                assert!(matches!(err, TellError::ResultCountMismatch { .. }));
            }
            _ => panic!("Expected Evaluate"),
        }
    }

    #[test]
    fn ask_tell_already_terminated() {
        let sampler = MetropolisHastings::new()
            .with_num_chains(1)
            .with_iterations(0); // Terminates immediately

        let mut state = sampler.init(vec![0.0]);

        // First ask should return Done (no iterations)
        match state.ask() {
            AskResult::Done(_) => (),
            _ => panic!("Expected Done on first ask"),
        }

        // Trying to tell should fail
        let err = state
            .tell(vec![Ok::<f64, std::io::Error>(1.0)])
            .unwrap_err();
        assert_eq!(err, TellError::AlreadyTerminated);
    }

    #[test]
    fn ask_tell_multiple_iterations() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| 0.5 * x[0].powi(2))
            .with_parameter("x", 1.0, Unbounded)
            .build()
            .unwrap();

        let sampler = MetropolisHastings::new()
            .with_num_chains(3)
            .with_iterations(10)
            .with_seed(123);

        let mut state = sampler.init(vec![0.5]);
        let mut iteration_count = 0;

        loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    assert_eq!(points.len(), 3, "Should have 3 proposals (one per chain)");
                    iteration_count += 1;
                    let results: Vec<_> = points.iter().map(|x| problem.evaluate(x)).collect();
                    state.tell(results).unwrap();
                }
                AskResult::Done(SamplingResults::MCMC(samples)) => {
                    assert_eq!(iteration_count, 10, "Should run exactly 10 iterations");
                    assert_eq!(
                        samples.draws(),
                        30,
                        "Should have 30 total draws (3 chains × 10 iterations)"
                    );
                    break;
                }
                _ => panic!("Expected MCMC results"),
            }
        }
    }

    #[test]
    fn ask_tell_handles_evaluation_errors() {
        let sampler = MetropolisHastings::new()
            .with_num_chains(2)
            .with_iterations(5)
            .with_seed(42);

        let mut state = sampler.init(vec![0.0]);
        let mut iteration = 0;

        loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    // Simulate some evaluation errors
                    let results: Vec<Result<f64, std::io::Error>> = points
                        .iter()
                        .enumerate()
                        .map(|(i, _)| {
                            if iteration == 2 && i == 0 {
                                Err(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    "Evaluation failed",
                                ))
                            } else {
                                Ok(1.0)
                            }
                        })
                        .collect();
                    state.tell(results).unwrap();
                    iteration += 1;
                }
                AskResult::Done(SamplingResults::MCMC(samples)) => {
                    // Should complete despite errors (errors treated as INFINITY)
                    assert_eq!(samples.chains().len(), 2);
                    assert_eq!(iteration, 5);
                    break;
                }
                _ => panic!("Expected MCMC results"),
            }
        }
    }
}
