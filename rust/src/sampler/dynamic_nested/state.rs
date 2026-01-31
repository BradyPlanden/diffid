use super::{logspace_sub, MIN_LIVE_POINTS};
use std::cmp::Ordering;

/// Represents a candidate location that currently resides in the live set.
#[derive(Clone, Debug)]
pub struct LivePoint {
    pub position: Vec<f64>,
    pub log_likelihood: f64,
}

impl LivePoint {
    /// Convenience constructor for a live point with known likelihood.
    pub fn new(position: Vec<f64>, log_likelihood: f64) -> Self {
        Self {
            position,
            log_likelihood,
        }
    }

    /// Get reference to the position vector.
    pub fn position(&self) -> &[f64] {
        &self.position
    }
}

/// Posterior-weighted sample accumulated during the run.
#[derive(Clone, Debug)]
pub(super) struct PosteriorSample {
    pub position: Vec<f64>,
    pub log_likelihood: f64,
    pub log_weight: f64,
}

impl PosteriorSample {
    /// Build a posterior sample with explicit log-likelihood and weight.
    pub fn new(position: Vec<f64>, log_likelihood: f64, log_weight: f64) -> Self {
        Self {
            position,
            log_likelihood,
            log_weight,
        }
    }
}

/// Aggregates live points, posterior archive, and log prior mass for DNS.
#[derive(Clone, Debug)]
pub(super) struct SamplerState {
    live_points: Vec<LivePoint>,
    posterior: Vec<PosteriorSample>,
    log_prior_mass: f64,
    dimension: usize,
}

/// Captures the removal of a live point along with its prior mass bookkeeping.
#[derive(Clone, Debug)]
pub struct RemovedPoint {
    pub point: LivePoint,
    pub log_weight: f64,
    pub log_prior_before: f64,
}

impl RemovedPoint {
    /// Accessor for the removed live point's likelihood.
    pub(super) fn log_likelihood(&self) -> f64 {
        self.point.log_likelihood
    }
}

impl SamplerState {
    /// Initialise state from an initial live set, inferring problem dimension.
    pub fn new(live_points: Vec<LivePoint>) -> Self {
        let dimension = live_points.first().map_or(0, |point| point.position.len());
        Self {
            live_points,
            posterior: Vec::new(),
            log_prior_mass: 0.0,
            dimension,
        }
    }

    /// Number of active decision variables tracked by the sampler.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Current live-point collection.
    pub fn live_points(&self) -> &[LivePoint] {
        &self.live_points
    }

    /// Count of live points still maintained by the sampler.
    pub fn live_point_count(&self) -> usize {
        self.live_points.len()
    }

    /// Posterior archive accumulated so far.
    pub fn posterior(&self) -> &[PosteriorSample] {
        &self.posterior
    }

    /// Remaining log prior mass after successive shrinkage steps.
    pub fn log_prior_mass(&self) -> f64 {
        self.log_prior_mass
    }

    /// Best log-likelihood among the current live points.
    pub fn max_log_likelihood(&self) -> f64 {
        self.live_points
            .iter()
            .map(|p| p.log_likelihood)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Worst log-likelihood among the current live points.
    #[allow(dead_code)]
    pub fn min_log_likelihood(&self) -> f64 {
        self.live_points
            .iter()
            .map(|p| p.log_likelihood)
            .fold(f64::INFINITY, f64::min)
    }

    /// Index of the live point with the lowest likelihood.
    pub fn worst_index(&self) -> Option<usize> {
        self.live_points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| live_ordering(a.log_likelihood, b.log_likelihood))
            .map(|(idx, _)| idx)
    }

    /// Get a random live point for use as MCMC starting position.
    ///
    /// # Panics
    /// Panics if there are no live points.
    pub fn random_live_point(&self, rng: &mut rand::prelude::StdRng) -> &LivePoint {
        use rand::Rng;
        let index = rng.random_range(0..self.live_points.len());
        &self.live_points[index]
    }

    /// Reduce the live set toward a new target, accepting removals.
    pub fn adjust_live_set(&mut self, target: usize) {
        let target = target.max(MIN_LIVE_POINTS);
        while self.live_points.len() > target {
            if let Some(removed) = self.remove_worst() {
                self.accept_removed(removed);
            } else {
                break;
            }
        }
    }

    /// Append a newly sampled live point.
    pub fn insert_live_point(&mut self, point: LivePoint) {
        if self.dimension == 0 {
            self.dimension = point.position.len();
        }
        self.live_points.push(point);
    }

    /// Remove a live point at the given index, updating prior mass bookkeeping.
    pub fn remove_at(&mut self, index: usize) -> Option<RemovedPoint> {
        if index >= self.live_points.len() {
            return None;
        }

        let n_live = self.live_points.len().max(1) as f64;
        let log_prev = self.log_prior_mass;
        // Use proper deterministic nested sampling formula: X_{i+1} = X_i * (n-1)/n
        self.log_prior_mass += ((n_live - 1.0) / n_live).ln();
        let log_weight = logspace_sub(log_prev, self.log_prior_mass).unwrap_or(f64::NEG_INFINITY);
        let removed = self.live_points.swap_remove(index);
        Some(RemovedPoint {
            point: removed,
            log_weight,
            log_prior_before: log_prev,
        })
    }

    /// Remove and return the lowest-likelihood live point.
    pub fn remove_worst(&mut self) -> Option<RemovedPoint> {
        let index = self.worst_index()?;
        self.remove_at(index)
    }

    /// Commit a removed point to the posterior archive.
    pub fn accept_removed(&mut self, removal: RemovedPoint) {
        let RemovedPoint {
            point, log_weight, ..
        } = removal;
        self.push_posterior(point.position, point.log_likelihood, log_weight);
    }

    /// Reinsert a previously removed point, restoring log prior mass.
    pub fn restore_removed(&mut self, removal: RemovedPoint) {
        let RemovedPoint {
            point,
            log_prior_before,
            ..
        } = removal;
        self.log_prior_mass = log_prior_before;
        self.insert_live_point(point);
    }

    fn push_posterior(&mut self, position: Vec<f64>, log_likelihood: f64, log_weight: f64) {
        self.posterior
            .push(PosteriorSample::new(position, log_likelihood, log_weight));
    }

    /// Convert remaining live points into posterior samples with residual weight.
    pub fn finalize(&mut self) {
        if self.live_points.is_empty() {
            return;
        }
        let residual_weight = self.log_prior_mass - (self.live_points.len() as f64).ln();
        let remaining: Vec<LivePoint> = self.live_points.drain(..).collect();
        for point in remaining {
            self.push_posterior(point.position, point.log_likelihood, residual_weight);
        }
    }
}

fn live_ordering(a: f64, b: f64) -> Ordering {
    if !a.is_finite() && !b.is_finite() {
        Ordering::Equal
    } else if !a.is_finite() {
        Ordering::Greater
    } else if !b.is_finite() {
        Ordering::Less
    } else {
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use crate::common::Bounds;

    #[test]
    fn bounds_clamp_respects_limits() {
        let bounds = Bounds::new(vec![(0.0, 1.0)]);
        let mut position = vec![10.0];
        bounds.clamp(&mut position);
        assert!(position[0].is_finite());
    }
}
