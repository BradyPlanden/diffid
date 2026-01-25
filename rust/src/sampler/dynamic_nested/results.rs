use super::state::PosteriorSample;
use crate::sampler::Samples;
use std::time::Duration;

/// Posterior sample with its likelihood and log-weight contribution.
#[derive(Clone, Debug)]
pub struct NestedSample {
    pub position: Vec<f64>,
    pub log_likelihood: f64,
    pub log_weight: f64,
}

impl NestedSample {
    /// Convenience helper returning the evidence contribution of this sample.
    pub fn evidence_weight(&self) -> f64 {
        if !self.log_likelihood.is_finite() || !self.log_weight.is_finite() {
            return 0.0;
        }
        (self.log_likelihood + self.log_weight).exp()
    }
}

/// Aggregated results of a Dynamic Nested Sampling run.
#[derive(Clone, Debug)]
pub struct NestedSamples {
    posterior: Vec<NestedSample>,
    mean: Vec<f64>,
    draws: usize,
    log_z: f64,
    information: f64,
    time: Duration,
}

impl NestedSamples {
    /// Build a [`NestedSamples`] view from raw posterior entries and problem dimension.
    pub(super) fn build(posterior: &[PosteriorSample], dimension: usize) -> Self {
        let mut samples = Vec::with_capacity(posterior.len());
        let mut log_z = f64::NEG_INFINITY;

        for sample in posterior {
            if !sample.log_likelihood.is_finite() || !sample.log_weight.is_finite() {
                continue;
            }

            let weight = sample.log_weight;
            let evidence = sample.log_likelihood + weight;
            log_z = logsumexp(log_z, evidence);
            samples.push(NestedSample {
                position: sample.position.clone(),
                log_likelihood: sample.log_likelihood,
                log_weight: weight,
            });
        }

        if samples.is_empty() {
            return Self::degenerate(vec![0.0; dimension.max(1)]);
        }

        let mut mean = vec![0.0; dimension.max(1)];
        let mut total_weight = f64::NEG_INFINITY;
        for sample in &samples {
            if !sample.log_weight.is_finite() || !sample.log_likelihood.is_finite() {
                continue;
            }

            let log_posterior_weight = sample.log_weight + sample.log_likelihood - log_z;
            if !log_posterior_weight.is_finite() {
                continue;
            }

            total_weight = logsumexp(total_weight, log_posterior_weight);
            let weight = log_posterior_weight.exp();
            for (i, value) in sample.position.iter().enumerate() {
                mean[i] += weight * value;
            }
        }

        if total_weight.is_finite() {
            let total = total_weight.exp().max(f64::MIN_POSITIVE);
            for value in &mut mean {
                *value /= total;
            }
        }

        let information = match information_from_samples(log_z, &samples) {
            info if info.is_finite() && info >= 0.0 => info,
            _ => 0.0,
        };

        Self {
            posterior: samples,
            mean,
            draws: posterior.len(),
            log_z,
            information,
            time: Duration::default(),
        }
    }

    /// Construct a result with no posterior support but a defined mean.
    pub fn degenerate(mean: Vec<f64>) -> Self {
        Self {
            posterior: Vec::new(),
            mean,
            draws: 0,
            log_z: f64::NEG_INFINITY,
            information: 0.0,
            time: Duration::default(),
        }
    }

    /// Record the elapsed execution time for the run.
    pub fn set_time(&mut self, time: Duration) {
        self.time = time;
    }

    /// Elapsed execution time for the run.
    pub fn time(&self) -> Duration {
        self.time
    }

    /// Posterior samples retained by the run.
    pub fn posterior(&self) -> &[NestedSample] {
        &self.posterior
    }

    /// Posterior mean estimated from normalized sample weights.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Number of posterior entries contributing to the result.
    pub fn draws(&self) -> usize {
        self.draws
    }

    /// Estimated log-evidence (log marginal likelihood).
    pub fn log_evidence(&self) -> f64 {
        self.log_z
    }

    /// Estimated information (Kullback–Leibler divergence) in nats.
    pub fn information(&self) -> f64 {
        self.information
    }

    /// Convert nested-sampling posterior into the generic [`Samples`] view.
    pub fn to_samples(&self) -> Samples {
        let chains = vec![self
            .posterior
            .iter()
            .map(|sample| sample.position.clone())
            .collect::<Vec<_>>()];

        // DynamicNested doesn't use MCMC so no acceptance data
        let acceptance_data = vec![Vec::new(); chains.len()];

        Samples::new(
            chains,
            self.mean.clone(),
            self.draws,
            self.time,
            acceptance_data,
        )
    }
}

/// Estimate nested-sampling information directly from posterior entries.
pub(super) fn information_estimate(posterior: &[PosteriorSample]) -> f64 {
    if posterior.is_empty() {
        return 0.0;
    }

    let mut log_z = f64::NEG_INFINITY;

    for sample in posterior {
        if !sample.log_likelihood.is_finite() || !sample.log_weight.is_finite() {
            continue;
        }
        let evidence = sample.log_likelihood + sample.log_weight;
        log_z = logsumexp(log_z, evidence);
    }

    let mut nested_samples = Vec::with_capacity(posterior.len());
    for sample in posterior {
        if !sample.log_likelihood.is_finite() || !sample.log_weight.is_finite() {
            continue;
        }
        nested_samples.push(NestedSample {
            position: Vec::new(),
            log_likelihood: sample.log_likelihood,
            log_weight: sample.log_weight,
        });
    }

    information_from_samples(log_z, &nested_samples)
}

/// Log-evidence accumulator for a slice of posterior samples.
pub(super) fn log_evidence_estimate(posterior: &[PosteriorSample]) -> f64 {
    let mut log_z = f64::NEG_INFINITY;
    for sample in posterior {
        if !sample.log_likelihood.is_finite() || !sample.log_weight.is_finite() {
            continue;
        }
        let evidence = sample.log_likelihood + sample.log_weight;
        log_z = logsumexp(log_z, evidence);
    }
    log_z
}

/// Helper computing the nested-sampling information (estimated KL divergence)
/// given log-evidence and pre-normalized sample entries.
fn information_from_samples(log_z: f64, samples: &[NestedSample]) -> f64 {
    if !log_z.is_finite() {
        return 0.0;
    }

    // Collect the unnormalized posterior log-weights and corresponding likelihoods.
    let mut deltas = Vec::with_capacity(samples.len());
    let mut values = Vec::with_capacity(samples.len());

    for sample in samples {
        if !sample.log_weight.is_finite() || !sample.log_likelihood.is_finite() {
            continue;
        }

        let delta = sample.log_weight + sample.log_likelihood - log_z;
        if !delta.is_finite() {
            continue;
        }

        deltas.push(delta);
        values.push(sample.log_likelihood);
    }

    if deltas.is_empty() {
        return 0.0;
    }

    // Shift by the maximum log-weight to maintain numerical stability.
    let max_delta = deltas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if !max_delta.is_finite() {
        return 0.0;
    }

    // Normalize the posterior weights via a log-sum-exp accumulation.
    let mut sum_exp = 0.0;
    for delta in &deltas {
        let term = (delta - max_delta).exp();
        if term.is_finite() {
            sum_exp += term;
        }
    }

    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return 0.0;
    }

    // Compute the log-normalization constant for the posterior weights.
    let log_norm = max_delta + sum_exp.ln();
    if !log_norm.is_finite() {
        return 0.0;
    }

    // Accumulate the KL divergence: sum w_i * (log L_i - log Z).
    // This is the core computation of the nested-sampling information.
    let mut information = 0.0;
    for (delta, log_likelihood) in deltas.iter().zip(values.iter()) {
        // Compute the posterior weight for this sample.
        let weight = (delta - log_norm).exp();
        if weight.is_finite() {
            // Accumulate the weighted log-likelihood excess.
            information += weight * (log_likelihood - log_z);
        }
    }

    // Return the maximum of the computed information and zero.
    information.max(0.0)
}

fn logsumexp(a: f64, b: f64) -> f64 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }

    let max = a.max(b);
    let min = a.min(b);

    if max.is_infinite() {
        return max;
    }

    max + (min - max).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn posterior_samples() -> Vec<PosteriorSample> {
        vec![
            PosteriorSample::new(vec![0.0], -1.0, 0.5f64.ln()),
            PosteriorSample::new(vec![1.0], -0.3, 0.8f64.ln()),
            PosteriorSample::new(vec![2.0], -0.1, 0.6f64.ln()),
        ]
    }

    #[test]
    fn nested_samples_preserves_chains_and_mean() {
        let posterior = posterior_samples();
        let nested = NestedSamples::build(&posterior, 1);
        let samples = nested.to_samples();

        assert_eq!(samples.chains().len(), 1);
        assert_eq!(samples.chains()[0].len(), posterior.len());
        assert_eq!(samples.draws(), posterior.len());
        assert_eq!(nested.mean().len(), 1);
        assert_eq!(samples.mean_x(), nested.mean());
    }

    #[test]
    fn evidence_weight_is_positive() {
        let nested = NestedSample {
            position: vec![0.0],
            log_likelihood: -0.3,
            log_weight: -1.0,
        };
        assert!(nested.evidence_weight() > 0.0);
    }

    #[test]
    fn logsumexp_handles_infinities() {
        assert_eq!(logsumexp(f64::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(logsumexp(0.0, f64::NEG_INFINITY), 0.0);
    }

    proptest! {
        #[test]
        fn to_samples_preserves_order_and_normalization(
            samples in proptest::collection::vec(
                (
                    proptest::collection::vec(-5.0..5.0, 1..4),
                    -10.0f64..0.0,
                    -8.0f64..2.0,
                ),
                1..6
            )
        ) {
            let dimension = samples[0].0.len();
            prop_assume!(dimension > 0);
            prop_assume!(samples.iter().all(|(pos, _, _)| pos.len() == dimension));

            let posterior: Vec<PosteriorSample> = samples
                .iter()
                .map(|(pos, ll, lw)| PosteriorSample::new(pos.clone(), *ll, *lw))
                .collect();

            let nested = NestedSamples::build(&posterior, dimension);
            prop_assume!(!nested.posterior().is_empty());

            let samples_view = nested.to_samples();
            let chains = samples_view.chains();
            let expected_positions: Vec<Vec<f64>> = samples.iter().map(|(pos, _, _)| pos.clone()).collect();
            prop_assert_eq!(chains[0].as_slice(), expected_positions.as_slice());

            let weights: Vec<f64> = nested
                .posterior()
                .iter()
                .map(|sample| (sample.log_weight + sample.log_likelihood - nested.log_evidence()).exp())
                .collect();

            let weight_sum: f64 = weights.iter().sum();
            prop_assume!(weight_sum.is_finite() && weight_sum > 0.0);
            prop_assert!((weight_sum - 1.0).abs() < 1e-2);
        }
    }
}
