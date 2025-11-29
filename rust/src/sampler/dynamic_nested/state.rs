use super::super::evaluate;
use super::logspace_sub;
use super::MIN_LIVE_POINTS;
use crate::problem::Problem;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::StandardNormal;
use std::cmp::Ordering;

/// Axis-aligned box defining sampling limits and proposal scales.
#[derive(Clone, Debug)]
pub(super) struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

const INITIAL_EVAL_BATCH_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ScalarProblemBuilder;
    use rand::SeedableRng;

    fn scalar_problem() -> Problem {
        ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| (x[0] - 1.0).powi(2))
            .build()
            .expect("failed to build test problem")
    }

    #[test]
    fn bounds_clamp_respects_limits() {
        let problem = scalar_problem();
        let bounds = Bounds::from_problem(&problem, &[0.0], 1);
        let mut position = vec![10.0];
        bounds.clamp(&mut position);
        assert!(position[0].is_finite());
    }

    #[test]
    fn sampler_state_removal_and_restoration() {
        let problem = scalar_problem();
        let mut rng = StdRng::seed_from_u64(123);
        let bounds = Bounds::from_problem(&problem, &[0.0], 1);
        let live_points = initial_live_points(&problem, &bounds, &mut rng, 16, 0.1, false);
        let mut state = SamplerState::new(live_points);

        let original_count = state.live_point_count();
        let removal = state.remove_worst().expect("expected removal");
        assert_eq!(state.live_point_count(), original_count - 1);

        state.restore_removed(removal);
        assert_eq!(state.live_point_count(), original_count);
    }

    #[test]
    fn initial_live_points_use_requested_count() {
        let problem = scalar_problem();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = Bounds::from_problem(&problem, &[0.0], 1);
        let live_points = initial_live_points(&problem, &bounds, &mut rng, 12, 0.1, false);
        assert_eq!(live_points.len(), 12);
    }
}

impl Bounds {
    /// Derive bounds from problem definitions or heuristics around the initial point.
    pub fn from_problem(problem: &Problem, initial: &[f64], dimension: usize) -> Self {
        let mut lower = vec![0.0; dimension];
        let mut upper = vec![0.0; dimension];

        for i in 0..dimension {
            if let Some(spec) = problem.parameter_specs().specs().get(i) {
                if let Some((lo, hi)) = spec.bounds {
                    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                    lower[i] = lo;
                    upper[i] = hi;
                    continue;
                }
            }

            let centre = initial.get(i).copied().unwrap_or(0.0);
            let scale = (centre.abs().max(1.0)) * 5.0;
            lower[i] = centre - scale;
            upper[i] = centre + scale;
        }

        Self { lower, upper }
    }

    /// Clamp a position in-place to remain inside the bounding box.
    pub fn clamp(&self, position: &mut [f64]) {
        for (value, (&lo, &hi)) in position
            .iter_mut()
            .zip(self.lower.iter().zip(self.upper.iter()))
        {
            if lo.is_finite() && *value < lo {
                *value = lo;
            }
            if hi.is_finite() && *value > hi {
                *value = hi;
            }
        }
    }

    /// Draw a random position inside the bounds with optional Gaussian expansion.
    pub fn sample(&self, rng: &mut StdRng, expansion_factor: f64) -> Vec<f64> {
        let mut position = Vec::with_capacity(self.lower.len());
        let scale = expansion_factor.abs().max(0.05);
        for (&lo, &hi) in self.lower.iter().zip(self.upper.iter()) {
            if lo.is_finite() && hi.is_finite() && lo < hi {
                let base = rng.random_range(lo..=hi);
                let width = hi - lo;
                let sigma = (width * scale).max(1e-6);
                let draw = rng.sample::<f64, _>(StandardNormal);
                let mut value = base + draw * sigma;
                if value < lo {
                    value = lo;
                } else if value > hi {
                    value = hi;
                }
                position.push(value);
            } else {
                let base = if lo.is_finite() {
                    lo
                } else if hi.is_finite() {
                    hi
                } else {
                    0.0
                };
                let sigma = scale.max(0.1);
                let offset = rng.sample::<f64, _>(StandardNormal);
                position.push(base + offset * sigma);
            }
        }
        position
    }

    /// Return the dimensionality implied by the bounds.
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }
}

/// Represents a candidate location that currently resides in the live set.
#[derive(Clone, Debug)]
pub(super) struct LivePoint {
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
pub(super) struct RemovedPoint {
    pub(super) point: LivePoint,
    pub(super) log_weight: f64,
    pub(super) log_prior_before: f64,
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
        let dimension = live_points
            .first()
            .map(|point| point.position.len())
            .unwrap_or(0);
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
        self.log_prior_mass += -1.0_f64 / n_live;
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

/// Generate an initial set of live points by sampling within bounds.
pub(super) fn initial_live_points(
    problem: &Problem,
    bounds: &Bounds,
    rng: &mut StdRng,
    live_points: usize,
    expansion_factor: f64,
    parallel: bool,
) -> Vec<LivePoint> {
    let mut samples = Vec::with_capacity(live_points);
    let mut attempts = 0usize;
    let max_attempts = live_points.saturating_mul(200).max(1000);

    if parallel {
        while samples.len() < live_points && attempts < max_attempts {
            let mut batch = Vec::with_capacity(INITIAL_EVAL_BATCH_SIZE);
            while samples.len().saturating_add(batch.len()) < live_points
                && attempts < max_attempts
                && batch.len() < INITIAL_EVAL_BATCH_SIZE
            {
                attempts = attempts.saturating_add(1);
                let mut position = bounds.sample(rng, expansion_factor);
                bounds.clamp(&mut position);
                batch.push(position);
            }

            if batch.is_empty() {
                continue;
            }

            let results = problem.evaluate_population(&batch);
            for (position, result) in batch.into_iter().zip(results.into_iter()) {
                let log_likelihood = match result {
                    Ok(value) => {
                        let ll = -value;
                        if ll.is_finite() {
                            ll
                        } else {
                            continue;
                        }
                    }
                    Err(_) => continue,
                };

                samples.push(LivePoint::new(position, log_likelihood));
                if samples.len() >= live_points {
                    break;
                }
            }
        }
    } else {
        while samples.len() < live_points && attempts < max_attempts {
            attempts += 1;
            let mut position = bounds.sample(rng, expansion_factor);
            bounds.clamp(&mut position);
            let log_likelihood = -evaluate(problem, &position);
            if !log_likelihood.is_finite() {
                continue;
            }
            samples.push(LivePoint::new(position, log_likelihood));
        }
    }

    samples
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
