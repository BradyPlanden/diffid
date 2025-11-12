use crate::problem::Problem;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

mod dynamic_nested;

pub use dynamic_nested::{DynamicNestedSampler, NestedSample, NestedSamples};

/// Core behaviour shared by all samplers.
pub trait Sampler {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> Samples;
}

#[derive(Clone, Debug)]
pub struct Samples {
    chains: Vec<Vec<Vec<f64>>>,
    mean_x: Vec<f64>,
    draws: usize,
}

impl Samples {
    pub fn new(chains: Vec<Vec<Vec<f64>>>, mean_x: Vec<f64>, draws: usize) -> Self {
        Self {
            chains,
            mean_x,
            draws,
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
}

#[derive(Clone, Debug)]
pub struct MetropolisHastings {
    num_chains: usize,
    parallel: bool,
    iterations: usize,
    step_size: f64,
    seed: Option<u64>,
}

impl MetropolisHastings {
    pub fn new() -> Self {
        Self {
            num_chains: 1,
            parallel: false,
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

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
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

impl Sampler for MetropolisHastings {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> Samples {
        let dimension = match (problem.dimension(), initial.len()) {
            (d, _) if d > 0 => d,
            (0, len) if len > 0 => len,
            _ => 1,
        };

        let mut start = if initial.is_empty() {
            vec![0.0; dimension]
        } else {
            let mut init = initial;
            if init.len() < dimension {
                init.resize(dimension, 0.0);
            } else if init.len() > dimension {
                init.truncate(dimension);
            }
            init
        };

        if start.len() != dimension {
            start.resize(dimension, 0.0);
        }

        let num_chains = self.num_chains.max(1);
        let iterations = self.iterations;
        let step_size = self.step_size;

        let mut seed_rng: StdRng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                StdRng::from_rng(rand::thread_rng()).unwrap_or_else(|_| StdRng::seed_from_u64(0))
            }
        };

        let seeds: Vec<u64> = (0..num_chains).map(|_| seed_rng.gen()).collect();
        let initial_state = start.clone();

        let chains: Vec<Vec<Vec<f64>>> = if self.parallel && num_chains > 1 {
            seeds
                .into_par_iter()
                .map(|seed| run_chain(problem, &initial_state, iterations, step_size, seed))
                .collect()
        } else {
            seeds
                .into_iter()
                .map(|seed| run_chain(problem, &initial_state, iterations, step_size, seed))
                .collect()
        };

        let draws = iterations.saturating_mul(num_chains);
        let mut mean_x = initial_state.clone();

        if draws > 0 {
            mean_x.fill(0.0);
            for chain in &chains {
                for sample in chain.iter().skip(1) {
                    for (i, value) in sample.iter().enumerate() {
                        mean_x[i] += *value;
                    }
                }
            }

            for value in &mut mean_x {
                *value /= draws as f64;
            }
        } else if let Some(last) = chains.first().and_then(|chain| chain.last()).cloned() {
            mean_x = last;
        }

        Samples::new(chains, mean_x, draws)
    }
}

fn run_chain(
    problem: &Problem,
    initial: &[f64],
    iterations: usize,
    step_size: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut current = initial.to_vec();
    let mut current_val = evaluate(problem, &current);
    let mut samples = Vec::with_capacity(iterations.saturating_add(1));
    samples.push(current.clone());

    for _ in 0..iterations {
        let mut proposal = current.clone();
        for value in &mut proposal {
            let noise: f64 = rng.sample(StandardNormal);
            *value += step_size * noise;
        }

        let proposal_val = evaluate(problem, &proposal);

        let accept = if !proposal_val.is_finite() {
            false
        } else {
            let acceptance_log = current_val - proposal_val;
            if acceptance_log >= 0.0 {
                true
            } else {
                let u: f64 = rng.gen();
                u < acceptance_log.exp()
            }
        };

        if accept {
            current = proposal;
            current_val = proposal_val;
        }

        samples.push(current.clone());
    }

    samples
}

pub(super) fn evaluate(problem: &Problem, x: &[f64]) -> f64 {
    match problem.evaluate(x) {
        Ok(value) => value,
        Err(_) => f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{BuilderParameterExt, ParameterSpec, ScalarProblemBuilder};

    #[test]
    fn metropolis_hastings_produces_samples() {
        let problem = ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| {
                let diff = x[0] - 1.0;
                0.5 * diff * diff
            })
            .with_parameter(ParameterSpec::new("x", 1.0, None))
            .build()
            .expect("problem to build");

        let sampler = MetropolisHastings::new()
            .with_num_chains(4)
            .with_iterations(600)
            .with_step_size(0.3)
            .with_seed(42);

        let samples = sampler.run(&problem, vec![0.0]);

        assert_eq!(samples.chains().len(), 4);
        for chain in samples.chains() {
            assert_eq!(chain.len(), 601);
        }

        let mean = samples.mean_x();
        assert_eq!(mean.len(), 1);
        assert!((mean[0] - 1.0).abs() < 0.2);
        assert_eq!(samples.draws(), 4 * 600);
    }
}
