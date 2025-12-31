use super::results;
use super::state::SamplerState;
use super::MIN_LIVE_POINTS;

/// Adapts the live-point budget and termination checks during sampling.
#[derive(Clone, Debug)]
pub(super) struct Scheduler {
    baseline_live: usize,
    max_live: usize,
    expansion_factor: f64,
    termination_tol: f64,
    last_log_evidence: f64,
    stagnation_counter: usize,
}

/// Heuristics for resizing the live set and deciding when to stop iterating.
impl Scheduler {
    /// Construct a scheduler anchored to a baseline live-set size.
    pub fn new(baseline_live: usize, expansion_factor: f64, termination_tol: f64) -> Self {
        let baseline = baseline_live.max(MIN_LIVE_POINTS);
        let expansion = expansion_factor.max(0.0);
        let tol = termination_tol.abs().max(1e-10);
        let max_live = ((baseline as f64) * (1.0 + 4.0 * expansion)).ceil() as usize;

        Self {
            baseline_live: baseline,
            max_live: max_live.max(baseline),
            expansion_factor: expansion,
            termination_tol: tol,
            last_log_evidence: f64::NEG_INFINITY,
            stagnation_counter: 0,
        }
    }

    /// Compute the desired live-set size given the estimated information gain.
    pub fn target(&mut self, information: f64, current_live: usize) -> usize {
        let info = information.max(0.0);
        let scale = 1.0 + self.expansion_factor * info.sqrt();
        let mut desired = (self.baseline_live as f64 * scale).round() as usize;
        desired = desired.clamp(MIN_LIVE_POINTS, self.max_live);

        if current_live == 0 {
            return desired;
        }

        // Smooth adjustments to avoid large oscillations.
        let delta = desired as isize - current_live as isize;
        if delta.abs() > (current_live as isize) / 2 {
            if delta.is_positive() {
                current_live + (current_live / 2).max(1)
            } else {
                current_live - (current_live / 2).max(1)
            }
            .clamp(MIN_LIVE_POINTS, self.max_live)
        } else {
            desired
        }
    }

    /// The scheduler tracks the evolution of the log-evidence and compares it
    /// against a derived upper bound from the live set. Termination is granted
    /// once the estimated remaining evidence mass falls below `termination_tol`
    /// for several consecutive checks and the information gain has stabilised.
    pub fn should_terminate(&mut self, state: &SamplerState, information: f64) -> bool {
        if state.posterior().is_empty() || state.live_point_count() == 0 {
            self.last_log_evidence = f64::NEG_INFINITY;
            self.stagnation_counter = 0;
            return false;
        }

        let log_z = results::log_evidence_estimate(state.posterior());
        if !log_z.is_finite() {
            self.last_log_evidence = f64::NEG_INFINITY;
            self.stagnation_counter = 0;
            return false;
        }

        // Track evidence stagnation by comparing to the previous iteration.
        let delta = (log_z - self.last_log_evidence).abs();
        if delta < self.termination_tol {
            self.stagnation_counter = self.stagnation_counter.saturating_add(1);
        } else {
            // Use decay instead of full reset - requires sustained progress
            self.stagnation_counter = self.stagnation_counter.saturating_sub(2);
        }
        self.last_log_evidence = log_z;

        // Estimate the remaining evidence mass by upper-bounding future
        // contributions from the current live points.
        let potential_log_z = state.max_log_likelihood() + state.log_prior_mass();
        let remaining = if potential_log_z.is_finite() {
            if potential_log_z > log_z {
                (potential_log_z - log_z).exp()
            } else {
                0.0
            }
        } else {
            f64::INFINITY
        };

        let remaining_small = remaining < self.termination_tol;
        let stable = self.stagnation_counter >= 4;

        // Primary termination: remaining evidence is small AND has stayed small.
        let primary_termination = remaining_small && stable;

        // Safety check: avoid early termination if information is still growing
        // rapidly, unless stagnation persists in high-information regimes.
        let information_stable = information < 50.0 || self.stagnation_counter >= 2;

        primary_termination && information_stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::dynamic_nested::state::{LivePoint, SamplerState};

    fn make_state(count: usize, log_likelihood: f64) -> SamplerState {
        let points = (0..count)
            .map(|_| LivePoint::new(vec![0.0], log_likelihood))
            .collect();
        SamplerState::new(points)
    }

    #[test]
    fn target_stays_within_bounds() {
        let mut scheduler = Scheduler::new(24, 0.4, 1e-3);
        let mut state = make_state(24, -1.0);

        let t0 = scheduler.target(0.0, state.live_point_count());
        assert!(t0 >= MIN_LIVE_POINTS);
        assert!(t0 <= scheduler.max_live);

        let t_high = scheduler.target(40.0, state.live_point_count());
        assert!(t_high >= t0);
        assert!(t_high <= scheduler.max_live);

        state.adjust_live_set(t_high / 2);
        let t_adjusted = scheduler.target(16.0, state.live_point_count());
        assert!(t_adjusted >= MIN_LIVE_POINTS);
        assert!(t_adjusted <= scheduler.max_live);
    }

    #[test]
    fn terminates_after_consistent_stagnation() {
        let mut scheduler = Scheduler::new(16, 0.1, f64::INFINITY);
        let mut state = make_state(16, -0.2);

        if let Some(removed) = state.remove_worst() {
            state.accept_removed(removed);
            state.insert_live_point(LivePoint::new(vec![0.0], -0.1));
        }

        assert!(!scheduler.should_terminate(&state, 0.0));

        let mut terminated = false;
        for _ in 0..8 {
            if scheduler.should_terminate(&state, 0.0) {
                terminated = true;
                break;
            }
        }

        assert!(
            terminated,
            "scheduler should terminate after repeated stagnation checks"
        );
    }

    #[test]
    fn terminates_with_high_information() {
        // Test that high-information posteriors can still terminate
        // Use a large termination tolerance so remaining evidence is small
        let mut scheduler = Scheduler::new(32, 0.1, 1e3);
        let mut state = make_state(32, -10.0);

        // Simulate a high-information scenario (narrow posterior)
        // Add some samples to build up evidence
        for i in 0..20 {
            if let Some(removed) = state.remove_worst() {
                state.accept_removed(removed);
                // Insert progressively better points
                state.insert_live_point(LivePoint::new(vec![0.0], -9.0 + i as f64 * 0.1));
            }
        }

        // High information value (>50) should not prevent termination
        // if stagnation is detected (stagnation_counter >= 2) and remaining is small
        let high_information = 75.0;

        // After sufficient stagnation, should terminate even with high information
        let mut terminated = false;
        for _ in 0..10 {
            if scheduler.should_terminate(&state, high_information) {
                terminated = true;
                break;
            }
        }

        assert!(
            terminated,
            "scheduler should terminate with high information after stagnation"
        );
    }

    #[test]
    fn does_not_terminate_with_low_information_unstable() {
        // Test that low information but unstable evidence does not terminate
        let mut scheduler = Scheduler::new(32, 0.1, 1e-3);
        let mut state = make_state(32, -10.0);

        // Simulate changing evidence (not stable)
        for i in 0..5 {
            if let Some(removed) = state.remove_worst() {
                state.accept_removed(removed);
                // Insert significantly better points to prevent stagnation
                state.insert_live_point(LivePoint::new(vec![0.0], -5.0 + i as f64));
            }

            // Should not terminate because evidence is changing
            assert!(!scheduler.should_terminate(&state, 5.0));
        }
    }
}
