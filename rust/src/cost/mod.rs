use diffsol::{MatrixCommon, NalgebraMat};
use std::f64::consts::PI;

/// Trait for cost metrics applied to residuals between simulated and observed data.
pub trait CostMetric: Send + Sync {
    fn evaluate(&self, residuals: &[f64]) -> f64;
    fn name(&self) -> &'static str;

    fn evaluate_with_sensitivities(
        &self,
        _residuals: &[f64],
        _sensitivities: &[NalgebraMat<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SumSquaredError {
    weight: f64,
}

impl Default for SumSquaredError {
    fn default() -> Self {
        Self { weight: 1.0 }
    }
}
impl SumSquaredError {
    pub fn new(weight: Option<f64>) -> Self {
        Self {
            weight: weight.unwrap_or(1.0),
        }
    }
}

impl CostMetric for SumSquaredError {
    #[inline]
    fn evaluate(&self, residuals: &[f64]) -> f64 {
        residuals.iter().map(|&r| r * r).sum::<f64>() * self.weight
    }

    fn name(&self) -> &'static str {
        "sse"
    }

    fn evaluate_with_sensitivities(
        &self,
        residuals: &[f64],
        sensitivities: &[NalgebraMat<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        let cost = self.evaluate(residuals);

        if sensitivities.is_empty() {
            return Some((cost, Vec::new()));
        }

        let num_params = sensitivities.len();
        let mut gradient = vec![0.0; num_params];

        for (param_idx, sens) in sensitivities.iter().enumerate() {
            let total_elements = sens.nrows() * sens.ncols();
            assert_eq!(
                total_elements,
                residuals.len(),
                "sensitivity matrix must have the same number of elements as residuals",
            );

            // Iterate through matrix elements in row-major order
            let mut dot = 0.0;
            let mut residual_idx = 0;
            for row in 0..sens.nrows() {
                for col in 0..sens.ncols() {
                    dot += 2.0 * residuals[residual_idx] * sens[(row, col)];
                    residual_idx += 1;
                }
            }

            gradient[param_idx] = dot * self.weight;
        }

        Some((cost, gradient))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RootMeanSquaredError {
    weight: f64,
}

impl Default for RootMeanSquaredError {
    fn default() -> Self {
        Self { weight: 1.0 }
    }
}
impl RootMeanSquaredError {
    pub fn new(weight: Option<f64>) -> Self {
        Self {
            weight: weight.unwrap_or(1.0),
        }
    }
}

impl CostMetric for RootMeanSquaredError {
    fn evaluate(&self, residuals: &[f64]) -> f64 {
        if residuals.is_empty() {
            return 0.0;
        }

        let n = residuals.len() as f64;
        let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / n;
        mse.sqrt() * self.weight
    }

    fn name(&self) -> &'static str {
        "rmse"
    }

    fn evaluate_with_sensitivities(
        &self,
        residuals: &[f64],
        sensitivities: &[NalgebraMat<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        if residuals.is_empty() {
            return Some((0.0, Vec::new()));
        }

        let n = residuals.len() as f64;
        let sse: f64 = residuals.iter().map(|&r| r * r).sum();
        let mse = sse / n;
        let rmse = mse.sqrt();

        if sensitivities.is_empty() {
            return Some((rmse * self.weight, Vec::new()));
        }

        let mut grad_rmse = vec![0.0; sensitivities.len()];

        for (param_idx, sens) in sensitivities.iter().enumerate() {
            let total_elements = sens.nrows() * sens.ncols();
            assert_eq!(
                total_elements,
                residuals.len(),
                "sensitivity matrix must have the same number of elements as residuals",
            );

            // Iterate through matrix elements in row-major order
            let mut dot = 0.0;
            let mut residual_idx = 0;
            for row in 0..sens.nrows() {
                for col in 0..sens.ncols() {
                    dot += 2.0 * residuals[residual_idx] * sens[(row, col)];
                    residual_idx += 1;
                }
            }
            grad_rmse[param_idx] = dot;
        }

        // Gradient: d(rmse)/dp = d(sqrt(mse))/dp = (1/(2*sqrt(mse))) * d(mse)/dp
        let gradient = if rmse > f64::EPSILON {
            grad_rmse
                .into_iter()
                .map(|g| g / (2.0 * n * rmse) * self.weight)
                .collect()
        } else {
            vec![0.0; sensitivities.len()]
        };

        Some((rmse * self.weight, gradient))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GaussianNll {
    weight: f64,
    variance: f64,
    log_term: f64, // Cache the constant term
}

impl Default for GaussianNll {
    fn default() -> Self {
        Self {
            weight: 1.0,
            variance: 1.0,
            log_term: 0.0,
        }
    }
}

impl GaussianNll {
    /// Creates a new Gaussian negative log-likelihood cost metric.
    ///
    /// # Arguments
    /// * `variance` - The variance of the Gaussian distribution (must be positive)
    ///
    /// # Panics
    /// Panics if variance is not positive and finite
    pub fn new(weight: Option<f64>, variance: f64) -> Self {
        assert!(
            variance > 0.0 && variance.is_finite(),
            "Variance must be positive and finite, got {variance}"
        );

        let log_term = (2.0 * PI * variance).ln();
        Self {
            weight: weight.unwrap_or(1.0),
            variance,
            log_term,
        }
    }

    /// Creates a new Gaussian NLL with variance clamped to a valid range.
    /// Use this when you want to handle invalid inputs gracefully.
    pub fn new_clamped(variance: f64) -> Self {
        let clamped = variance.clamp(f64::EPSILON, f64::MAX);
        Self::new(Some(1.0), clamped)
    }
}

impl CostMetric for GaussianNll {
    fn evaluate(&self, residuals: &[f64]) -> f64 {
        if residuals.is_empty() {
            return 0.0;
        }

        let n = residuals.len() as f64;
        let sse: f64 = residuals.iter().map(|&r| r * r).sum();

        // NLL = (n/2) * ln(2πσ²) + (1/2σ²) * Σr²
        (0.5 * n).mul_add(self.log_term, 0.5 * sse / self.variance) * self.weight
    }

    fn name(&self) -> &'static str {
        "gaussian_nll"
    }

    fn evaluate_with_sensitivities(
        &self,
        residuals: &[f64],
        sensitivities: &[NalgebraMat<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        if residuals.is_empty() {
            return Some((0.0, Vec::new()));
        }

        let n = residuals.len() as f64;
        let sse: f64 = residuals.iter().map(|&r| r * r).sum();

        let cost = (0.5 * n).mul_add(self.log_term, 0.5 * sse / self.variance) * self.weight;

        if sensitivities.is_empty() {
            return Some((cost, Vec::new()));
        }

        let mut grad_nll = vec![0.0; sensitivities.len()];

        for (param_idx, sens) in sensitivities.iter().enumerate() {
            let total_elements = sens.nrows() * sens.ncols();
            assert_eq!(
                total_elements,
                residuals.len(),
                "sensitivity matrix must have the same number of elements as residuals",
            );

            // Iterate through matrix elements in row-major order
            let mut dot = 0.0;
            let mut residual_idx = 0;
            for row in 0..sens.nrows() {
                for col in 0..sens.ncols() {
                    dot += residuals[residual_idx] * sens[(row, col)];
                    residual_idx += 1;
                }
            }
            grad_nll[param_idx] = dot;
        }

        // Gradient: d(NLL)/dp = d(NLL)/dy * dy/dp
        // Gradient: d(NLL)/dp = residual/variance * sensitivities
        let gradient: Vec<f64> = grad_nll
            .into_iter()
            .map(|g| g / self.variance * self.weight)
            .collect();

        Some((cost, gradient))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffsol::{Matrix, NalgebraMat};

    #[test]
    fn test_sse_basic() {
        let metric = SumSquaredError::new(None);
        let residuals = vec![1.0, 2.0, 3.0];
        assert_eq!(metric.evaluate(&residuals), 14.0);
    }

    #[test]
    fn test_sse_with_weight() {
        let metric = SumSquaredError::new(Some(2.0));
        let residuals = vec![1.0, 2.0, 3.0];
        // Unweighted SSE is 14.0, so with weight 2.0 it should be 28.0
        assert_eq!(metric.evaluate(&residuals), 28.0);
    }

    #[test]
    fn test_sse_with_gradient() {
        let metric = SumSquaredError::new(None);
        let residuals = vec![1.0, 2.0];

        // Create a 2x1 sensitivity matrix with values [0.5, 0.5]
        let triplets = vec![(0, 0, 0.5), (1, 0, 0.5)];
        let sens_matrix: NalgebraMat<f64> =
            Matrix::try_from_triplets(2, 1, triplets, Default::default()).unwrap();

        let (cost, grad) = metric
            .evaluate_with_sensitivities(&residuals, &[sens_matrix])
            .expect("SumSquaredError should support gradient evaluation");
        assert_eq!(cost, 5.0);
        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0], 3.0); // 2*1*0.5 + 2*2*0.5
    }

    #[test]
    fn test_rmse_basic() {
        let metric = RootMeanSquaredError::new(None);
        let residuals = vec![1.0, 2.0, 3.0];
        let expected = (14.0 / 3.0_f64).sqrt();
        assert!((metric.evaluate(&residuals) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_with_weight() {
        let metric = RootMeanSquaredError::new(Some(0.5));
        let residuals = vec![1.0, 2.0, 3.0];
        let expected = (14.0 / 3.0_f64).sqrt() * 0.5;
        assert!((metric.evaluate(&residuals) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_with_gradient() {
        let metric = RootMeanSquaredError::new(None);
        let residuals = vec![1.0, 2.0];

        // Create a 2x1 sensitivity matrix with values [0.5, 0.5]
        let triplets = vec![(0, 0, 0.5), (1, 0, 0.5)];
        let sens_matrix: NalgebraMat<f64> =
            Matrix::try_from_triplets(2, 1, triplets, Default::default()).unwrap();

        let (cost, grad) = metric
            .evaluate_with_sensitivities(&residuals, &[sens_matrix])
            .expect("RootMeanSquaredError should support gradient evaluation");

        // RMSE = sqrt((1 + 4) / 2) = sqrt(2.5)
        let expected_rmse = (2.5_f64).sqrt();
        assert!((cost - expected_rmse).abs() < 1e-10);
        assert_eq!(grad.len(), 1);

        // d(RMSE)/dp = (1/(2*n*rmse)) * sum(2 * r * dr/dp)
        // = (1/(2*2*sqrt(2.5))) * (2*1*0.5 + 2*2*0.5)
        // = (1/(4*sqrt(2.5))) * 3.0
        let expected_grad = 3.0 / (4.0 * expected_rmse);
        assert!((grad[0] - expected_grad).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Variance must be positive")]
    fn test_gaussian_nll_invalid_variance() {
        GaussianNll::new(None, -1.0);
    }

    #[test]
    fn test_gaussian_nll_clamped() {
        let metric = GaussianNll::new_clamped(-1.0);
        assert!(metric.variance > 0.0);
    }

    #[test]
    fn test_gaussian_nll_with_weight() {
        let residuals = vec![1.0, -1.0];
        let variance = 0.5;
        let base = GaussianNll::new(Some(1.0), variance).evaluate(&residuals);
        let weighted = GaussianNll::new(Some(2.0), variance).evaluate(&residuals);
        assert!((weighted - 2.0 * base).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_nll_with_gradient() {
        let variance = 1.0;
        let metric = GaussianNll::new(None, variance);
        let residuals = vec![1.0, 2.0];

        // Create a 2x1 sensitivity matrix with values [0.5, 0.5]
        let triplets = vec![(0, 0, 0.5), (1, 0, 0.5)];
        let sens_matrix: NalgebraMat<f64> =
            Matrix::try_from_triplets(2, 1, triplets, Default::default()).unwrap();

        let (cost, grad) = metric
            .evaluate_with_sensitivities(&residuals, &[sens_matrix])
            .expect("GaussianNll should support gradient evaluation");

        // NLL = (n/2) * ln(2πσ²) + (1/2σ²) * Σr²
        let n = 2.0;
        let sse = 1.0 + 4.0;
        let expected_cost = 0.5 * n * (2.0 * PI * variance).ln() + 0.5 * sse / variance;
        assert!((cost - expected_cost).abs() < 1e-10);

        assert_eq!(grad.len(), 1);

        // d(NLL)/dp = (1/variance) * sum(r * dr/dp)
        // = (1/1.0) * (1.0*0.5 + 2.0*0.5)
        // = 1.5
        let expected_grad = (1.0 * 0.5 + 2.0 * 0.5) / variance;
        assert!((grad[0] - expected_grad).abs() < 1e-10);
    }
}
