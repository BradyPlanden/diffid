use super::state::{Bounds, LivePoint};
use super::{evaluate, MIN_LIVE_POINTS};
use crate::problem::Problem;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::StandardNormal;

const MAX_ATTEMPTS_FACTOR: usize = 64;
const MAX_UNIFORM_ATTEMPTS: usize = 32;

/// Generates new live-point proposals using local perturbations and periodic uniform draws.
#[derive(Clone, Debug)]
pub(super) struct ProposalEngine {
    dimension: usize,
    expansion_factor: f64,
}

impl ProposalEngine {
    /// Instantiate a proposal engine with dimensionality-aware scales.
    pub fn new(dimension: usize, expansion_factor: f64) -> Self {
        Self {
            dimension: dimension.max(1),
            expansion_factor: expansion_factor.max(0.05),
        }
    }

    /// Sample a new live point above the given likelihood threshold, if possible.
    pub fn draw(
        &mut self,
        rng: &mut StdRng,
        problem: &Problem,
        live_points: &[LivePoint],
        bounds: &Bounds,
        threshold: f64,
        _parallel: bool,
    ) -> Option<LivePoint> {
        if live_points.is_empty() {
            return None;
        }

        let step_sizes = compute_scales(live_points, bounds, self.expansion_factor);
        let max_attempts = MAX_ATTEMPTS_FACTOR.saturating_mul(self.dimension).max(64);

        for attempt in 0..max_attempts {
            // Periodically try a fresh uniform draw in the bounding box to avoid stagnation.
            let candidate = if attempt % MAX_UNIFORM_ATTEMPTS == 0 {
                let mut position = bounds.sample(rng, self.expansion_factor);
                bounds.clamp(&mut position);
                position
            } else {
                match live_points.choose(rng) {
                    Some(anchor) => {
                        let mut proposal = anchor.position.clone();
                        for (value, scale) in proposal.iter_mut().zip(step_sizes.iter()) {
                            let perturb = rng.sample::<f64, _>(StandardNormal) * scale;
                            *value += perturb;
                        }
                        bounds.clamp(&mut proposal);
                        proposal
                    }
                    None => continue,
                }
            };

            let log_likelihood = -evaluate(problem, &candidate);
            if !log_likelihood.is_finite() {
                continue;
            }

            if log_likelihood > threshold || live_points.len() < MIN_LIVE_POINTS {
                return Some(LivePoint::new(candidate, log_likelihood));
            }
        }

        None
    }
}

/// Compute Gaussian perturbation scales per dimension using the spread of the
/// current live points, widened by the configured expansion factor. Falls back
/// to a uniform, expansion-factor-based scale when live points are unavailable
/// or bounds are degenerate.
fn compute_scales(live_points: &[LivePoint], bounds: &Bounds, expansion_factor: f64) -> Vec<f64> {
    let dimension = bounds.dimension();
    if live_points.is_empty() {
        return vec![expansion_factor.max(0.1); dimension];
    }

    let mut mins = vec![f64::INFINITY; dimension];
    let mut maxs = vec![f64::NEG_INFINITY; dimension];

    for point in live_points {
        for (i, value) in point.position.iter().enumerate() {
            mins[i] = mins[i].min(*value);
            maxs[i] = maxs[i].max(*value);
        }
    }

    mins.iter_mut().zip(maxs.iter_mut()).for_each(|(min, max)| {
        if !min.is_finite() {
            *min = -1.0;
        }
        if !max.is_finite() {
            *max = 1.0;
        }
        if *max <= *min {
            *max = *min + 1e-3;
        }
    });

    mins.iter()
        .zip(maxs.iter())
        .map(|(&min, &max)| {
            let width = (max - min).abs().max(1e-3);
            width * expansion_factor.max(0.05)
        })
        .collect()
}
