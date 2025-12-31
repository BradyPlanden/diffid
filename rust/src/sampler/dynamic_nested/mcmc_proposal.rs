use crate::common::Bounds;
use rand::prelude::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// MCMC proposal generator for Dynamic Nested Sampling
///
/// Generates batches of proposal points using a Gaussian random walk
/// starting from a given live point. The proposals form a chain
/// that explores the parameter space near the starting point.
///
/// This approach provides threshold-aware proposals for nested sampling:
/// - Start from a random live point (guaranteed to be above current threshold)
/// - Generate perturbations using Gaussian random walk
/// - Step sizes adapt to bounds width per dimension
/// - Returns batch of proposals for evaluation
#[derive(Debug, Clone)]
pub(super) struct MCMCProposalGenerator {
    /// Step size factor relative to bounds width per dimension
    step_size_factor: f64,
}

impl MCMCProposalGenerator {
    /// Create a new MCMC proposal generator
    ///
    /// # Arguments
    /// * `step_size_factor` - Relative step size (fraction of bounds width)
    ///   Should be > 0. Typical values: 0.05 to 0.2
    ///   Smaller values: more conservative exploration
    ///   Larger values: more aggressive exploration
    pub fn new(step_size_factor: f64) -> Self {
        Self {
            step_size_factor: step_size_factor.max(1e-6),
        }
    }

    /// Generate a batch of MCMC proposals via Gaussian perturbations
    ///
    /// Creates independent proposals by applying small Gaussian perturbations
    /// to the starting point. Uses adaptive step sizes based on bounds width.
    ///
    /// Note: For nested sampling, proposals should have likelihood > threshold.
    /// Small step sizes help ensure proposals stay near the high-likelihood
    /// region of the starting point (which is already above threshold).
    ///
    /// # Arguments
    /// * `start_point` - Starting position (typically a random live point)
    /// * `bounds` - Parameter bounds for clamping
    /// * `rng` - Random number generator
    /// * `batch_size` - Number of proposals to generate
    ///
    /// # Returns
    /// Vector of `batch_size` proposal points, all clamped to bounds
    pub fn generate_batch(
        &self,
        start_point: &[f64],
        bounds: &Bounds,
        rng: &mut StdRng,
        batch_size: usize,
    ) -> Vec<Vec<f64>> {
        let dim = start_point.len();
        let mut proposals = Vec::with_capacity(batch_size);

        // Calculate per-dimension step sizes based on bounds width
        let scales = self.compute_step_sizes(bounds);

        // Generate independent proposals from start point
        for _ in 0..batch_size {
            let mut proposal = start_point.to_vec();

            // Apply small Gaussian perturbation to each dimension
            for i in 0..dim {
                let noise: f64 = StandardNormal.sample(rng);
                proposal[i] += scales[i] * noise;
            }

            // Clamp to bounds
            bounds.clamp(&mut proposal);

            // Add to batch
            proposals.push(proposal);
        }

        proposals
    }

    /// Compute per-dimension step sizes based on bounds width
    ///
    /// For each dimension i, calculates:
    ///   scale[i] = step_size_factor × (upper[i] - lower[i])
    ///
    /// For unbounded dimensions, uses a default scale of 1.0.
    /// This makes step sizes adaptive to the natural scale of each parameter.
    fn compute_step_sizes(&self, bounds: &Bounds) -> Vec<f64> {
        bounds
            .limits
            .iter()
            .map(|range| {
                let width = range.end() - range.start();
                if width.is_finite() {
                    self.step_size_factor * width
                } else {
                    // For unbounded dimensions, use a reasonable default scale
                    1.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_proposals_within_bounds() {
        let bounds = Bounds::new(vec![(-10.0, 10.0), (-5.0, 5.0)]);
        let start = vec![0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(42);

        let generator = MCMCProposalGenerator::new(0.1);
        let proposals = generator.generate_batch(&start, &bounds, &mut rng, 10);

        assert_eq!(proposals.len(), 10);

        for proposal in proposals {
            assert_eq!(proposal.len(), 2);
            assert!(proposal[0] >= -10.0 && proposal[0] <= 10.0);
            assert!(proposal[1] >= -5.0 && proposal[1] <= 5.0);
        }
    }

    #[test]
    fn test_chain_explores_space() {
        let bounds = Bounds::new(vec![(-10.0, 10.0)]);
        let start = vec![0.0];
        let mut rng = StdRng::seed_from_u64(123);

        let generator = MCMCProposalGenerator::new(0.2);
        let proposals = generator.generate_batch(&start, &bounds, &mut rng, 20);

        // At least some proposals should differ from start point
        let moved = proposals.iter().any(|p| (p[0] - start[0]).abs() > 0.1);
        assert!(moved, "MCMC chain should explore away from start point");

        // Proposals should form a chain (each differs from neighbors)
        let mut distances = Vec::new();
        for i in 1..proposals.len() {
            let dist = (proposals[i][0] - proposals[i - 1][0]).abs();
            distances.push(dist);
        }

        // Average step distance should be non-zero
        let avg_dist: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
        assert!(avg_dist > 0.0, "Chain should be making steps");
    }

    #[test]
    fn test_step_size_scaling() {
        let bounds1 = Bounds::new(vec![(-1.0, 1.0)]);
        let bounds2 = Bounds::new(vec![(-10.0, 10.0)]);
        let start = vec![0.0];
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let generator = MCMCProposalGenerator::new(0.1);

        let proposals1 = generator.generate_batch(&start, &bounds1, &mut rng1, 10);
        let proposals2 = generator.generate_batch(&start, &bounds2, &mut rng2, 10);

        // Calculate average distance traveled
        let dist1: f64 = proposals1
            .iter()
            .map(|p| (p[0] - start[0]).abs())
            .sum::<f64>()
            / 10.0;
        let dist2: f64 = proposals2
            .iter()
            .map(|p| (p[0] - start[0]).abs())
            .sum::<f64>()
            / 10.0;

        // With same RNG seed, larger bounds should yield proportionally larger steps
        // Since bounds2 is 10x wider than bounds1, steps should be ~10x larger
        assert!(
            dist2 > dist1 * 5.0,
            "Larger bounds should produce larger steps"
        );
    }
}
