use crate::common::{AskResult, Bounds, Point};
use crate::errors::{EvaluationError, TellError};
use crate::optimisers::ScalarEvaluation;
use crate::prelude::{Samples, SamplingResults};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct MetropolisHastings {
    num_chains: usize,
    iterations: usize,
    step_size: f64,
    seed: Option<u64>,
}

impl MetropolisHastings {
    pub fn new() -> Self {
        Self {
            num_chains: 1,
            iterations: 1_000,
            step_size: 0.1,
            seed: None,
        }
    }

    pub fn with_num_chains(mut self, num_chains: usize) -> Self {
        self.num_chains = num_chains.max(1);
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size.abs().max(f64::MIN_POSITIVE);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for MetropolisHastings {
    fn default() -> Self {
        Self::new()
    }
}

impl MetropolisHastings {
    /// Run Metropolis-Hastings MCMC sampling
    ///
    /// This method evaluates the problem's objective function to sample
    /// from the posterior distribution using a random walk Metropolis algorithm.
    ///
    /// Internally uses the ask/tell interface for consistency. For external
    /// control of the evaluation loop, use `init()`, `ask()`, and `tell()` directly.
    pub fn run<F, R, E>(&self, mut objective: F, initial: Point, bounds: Bounds) -> Samples
    where
        F: FnMut(&[f64]) -> R,
        R: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        let initial_point = initial;
        let mut state = self.init(initial_point, bounds);

        loop {
            match state.ask() {
                AskResult::Evaluate(points) => {
                    let results: Vec<_> = points.iter().map(|x| objective(x)).collect();
                    state
                        .tell(results)
                        .expect("tell() should succeed during run()");
                }
                AskResult::Done(SamplingResults::MCMC(samples)) => {
                    return samples;
                }
                _ => unreachable!("MetropolisHastings always returns MCMC results"),
            }
        }
    }
}

/// State for ask/tell interface of Metropolis-Hastings sampler
pub struct MetropolisHastingsState {
    chains: Vec<ChainState>,
    step_size: f64,
    max_iterations: usize,
    iteration: usize,
    start_time: Instant,
    phase: MCMCPhase,
    bounds: Bounds,
}

/// Internal state for a single MCMC chain
struct ChainState {
    current: Vec<f64>,
    current_log_likelihood: f64,
    proposal: Vec<f64>,
    samples: Vec<Vec<f64>>,
    rng: StdRng,
}

/// Phase tracking for MCMC sampling
enum MCMCPhase {
    /// Waiting for proposal evaluations
    AwaitingProposals,
    /// Sampling has terminated
    Terminated,
}

impl MetropolisHastings {
    /// Initialize ask/tell state for external control of sampling
    ///
    /// This allows you to control the evaluation loop externally, enabling
    /// parallel evaluation, distributed computing, or custom evaluation strategies.
    ///
    /// # Arguments
    /// * `initial` - Starting point for all chains
    ///
    /// # Returns
    /// Initial state ready for the first `ask()` call
    ///
    /// # Example
    /// ```ignore
    /// let sampler = MetropolisHastings::new().with_num_chains(4);
    /// let mut state = sampler.init(vec![0.0]);
    ///
    /// loop {
    ///     match state.ask() {
    ///         AskResult::Evaluate(points) => {
    ///             let results = points.iter().map(|x| problem.evaluate(x)).collect();
    ///             state.tell(results).unwrap();
    ///         }
    ///         AskResult::Done(results) => break,
    ///     }
    /// }
    /// ```
    pub fn init(&self, initial: Vec<f64>, bounds: Bounds) -> MetropolisHastingsState {
        let mut seed_rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        // Move initial and clamp it
        let mut initial_point = initial;
        bounds.clamp(&mut initial_point);

        let chains: Vec<ChainState> = (0..self.num_chains)
            .map(|_| {
                let mut rng = StdRng::seed_from_u64(seed_rng.random());
                let mut proposal = initial_point.clone();
                for val in &mut proposal {
                    let noise: f64 = rng.sample(StandardNormal);
                    *val += self.step_size * noise;
                }

                bounds.clamp(&mut proposal);
                ChainState {
                    current: initial_point.clone(),
                    current_log_likelihood: f64::INFINITY,
                    proposal,
                    samples: vec![initial_point.clone()],
                    rng,
                }
            })
            .collect();

        MetropolisHastingsState {
            chains,
            step_size: self.step_size,
            max_iterations: self.iterations,
            iteration: 0,
            start_time: Instant::now(),
            phase: MCMCPhase::AwaitingProposals,
            bounds,
        }
    }
}

impl MetropolisHastingsState {
    /// Get next point(s) to evaluate
    ///
    /// Returns either:
    /// - `Evaluate(points)`: Evaluate these proposal points and call `tell()` with results
    /// - `Done(results)`: Sampling complete, contains final `SamplingResults`
    ///
    /// # Returns
    /// `AskResult` indicating next action or final results
    pub fn ask(&self) -> AskResult<SamplingResults> {
        match self.phase {
            MCMCPhase::Terminated => AskResult::Done(SamplingResults::MCMC(self.build_results())),
            MCMCPhase::AwaitingProposals => {
                let proposals: Vec<Vec<f64>> = self
                    .chains
                    .iter()
                    .map(|chain| chain.proposal.clone())
                    .collect();
                AskResult::Evaluate(proposals)
            }
        }
    }

    /// Report evaluation results for proposed points
    ///
    /// Provide the negative log-likelihood (or objective value) for each proposal
    /// returned by the last `ask()` call. The number of results must match the
    /// number of proposals.
    ///
    /// # Arguments
    /// * `results` - Evaluation results (one per proposal point)
    ///
    /// # Returns
    /// `Ok(())` if successful, or `TellError` if:
    /// - Already terminated
    /// - Wrong number of results
    ///
    /// # Example
    /// ```ignore
    /// match state.ask() {
    ///     AskResult::Evaluate(points) => {
    ///         let results: Vec<_> = points
    ///             .iter()
    ///             .map(|x| problem.evaluate(x))
    ///             .collect();
    ///         state.tell(results)?;
    ///     }
    ///     AskResult::Done(results) => { /* done */ }
    /// }
    /// ```
    pub fn tell<I, T, E>(&mut self, results: I) -> Result<(), TellError>
    where
        I: IntoIterator<Item = T>,
        T: TryInto<ScalarEvaluation, Error = E>,
        E: Into<EvaluationError>,
    {
        if matches!(self.phase, MCMCPhase::Terminated) {
            return Err(TellError::AlreadyTerminated);
        }

        let values: Vec<f64> = results
            .into_iter()
            .map(|r| match r.try_into() {
                Ok(eval) => eval.value(),
                Err(_) => f64::INFINITY,
            })
            .collect();

        if values.len() != self.chains.len() {
            return Err(TellError::ResultCountMismatch {
                expected: self.chains.len(),
                got: values.len(),
            });
        }

        // Process each chain's accept/reject decision
        for (chain, value) in self.chains.iter_mut().zip(values) {
            let proposal_log_likelihood = value;

            // Metropolis-Hastings acceptance criterion
            let accept_log = chain.current_log_likelihood - proposal_log_likelihood;
            let accept = if accept_log >= 0.0 {
                true
            } else if proposal_log_likelihood.is_finite() {
                let u: f64 = chain.rng.random();
                u < accept_log.exp()
            } else {
                false
            };

            if accept {
                chain.current = chain.proposal.clone();
                chain.current_log_likelihood = proposal_log_likelihood;
            }

            chain.samples.push(chain.current.clone());

            // Generate next proposal
            chain.proposal = chain.current.clone();
            for val in &mut chain.proposal {
                let noise: f64 = chain.rng.sample(StandardNormal);
                *val += self.step_size * noise;
            }

            // Clamp the proposal
            self.bounds.clamp(&mut chain.proposal);
        }

        self.iteration += 1;

        if self.iteration >= self.max_iterations {
            self.phase = MCMCPhase::Terminated;
        }

        Ok(())
    }

    /// Get current iteration count
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// Get number of chains
    pub fn num_chains(&self) -> usize {
        self.chains.len()
    }

    /// Build final sampling results from accumulated chain data
    fn build_results(&self) -> Samples {
        let chains: Vec<Vec<Vec<f64>>> = self
            .chains
            .iter()
            .map(|chain| chain.samples.clone())
            .collect();

        let dimension = self.chains[0].current.len();
        let mut mean_x = vec![0.0; dimension];

        for chain in &self.chains {
            for sample in chain.samples.iter().skip(1) {
                for (i, &val) in sample.iter().enumerate() {
                    mean_x[i] += val;
                }
            }
        }

        let total_samples = self
            .chains
            .iter()
            .map(|c| c.samples.len().saturating_sub(1))
            .sum::<usize>();

        if total_samples > 0 {
            for val in &mut mean_x {
                *val /= total_samples as f64;
            }
        }

        Samples::new(chains, mean_x, total_samples, self.start_time.elapsed())
    }
}
