use nalgebra::DMatrix;
use std::f64::consts::PI;

/// Trait for cost metrics applied to residuals between simulated and observed data.
pub trait CostMetric: Send + Sync {
    fn evaluate(&self, residuals: &[f64]) -> f64;
    fn name(&self) -> &'static str;

    /// Optionally evaluate the cost and its gradient with respect to parameters.
    ///
    /// Implementations that support analytical gradients should override this and
    /// return `Some((cost, gradient))`. The default implementation returns
    /// `None`, indicating that gradient computation is not available.
    fn evaluate_with_sensitivities(
        &self,
        _residuals: &[f64],
        _sensitivities: &[DMatrix<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        None
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SumSquaredError;

impl CostMetric for SumSquaredError {
    #[inline]
    fn evaluate(&self, residuals: &[f64]) -> f64 {
        residuals.iter().map(|&r| r * r).sum()
    }

    fn name(&self) -> &'static str {
        "sse"
    }

    fn evaluate_with_sensitivities(
        &self,
        residuals: &[f64],
        sensitivities: &[DMatrix<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        let cost = self.evaluate(residuals);

        if sensitivities.is_empty() {
            return Some((cost, Vec::new()));
        }

        let num_params = sensitivities.len();
        let mut gradient = vec![0.0; num_params];

        for (param_idx, sens) in sensitivities.iter().enumerate() {
            assert_eq!(
                sens.len(),
                residuals.len(),
                "sensitivity matrix must have the same number of elements as residuals",
            );

            let dot: f64 = sens
                .iter()
                .zip(residuals.iter())
                .map(|(s, r)| 2.0 * r * s)
                .sum();
            gradient[param_idx] = dot;
        }

        Some((cost, gradient))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RootMeanSquaredError;

impl RootMeanSquaredError {
    fn evaluate_with_sensitivities_impl(
        &self,
        residuals: &[f64],
        sensitivities: &[DMatrix<f64>],
    ) -> (f64, Vec<f64>) {
        if residuals.is_empty() {
            return (0.0, Vec::new());
        }

        let n = residuals.len() as f64;
        let sse: f64 = residuals.iter().map(|&r| r * r).sum();
        let mse = sse / n;
        let rmse = mse.sqrt();

        if sensitivities.is_empty() {
            return (rmse, Vec::new());
        }

        let mut grad_rmse = vec![0.0; sensitivities.len()];

        for (param_idx, sens) in sensitivities.iter().enumerate() {
            assert_eq!(
                sens.len(),
                residuals.len(),
                "sensitivity matrix must have the same number of elements as residuals",
            );

            let dot: f64 = sens
                .iter()
                .zip(residuals.iter())
                .map(|(s, r)| 2.0 * r * s)
                .sum();
            grad_rmse[param_idx] = dot;
        }

        // Gradient: d(rmse)/dp = d(sqrt(mse))/dp = (1/(2*sqrt(mse))) * d(mse)/dp * dy/dp
        let gradient = if rmse > f64::EPSILON {
            grad_rmse
                .into_iter()
                .map(|g| g / (2.0 * n * rmse))
                .collect()
        } else {
            vec![0.0; sensitivities.len()]
        };

        (rmse, gradient)
    }
}

impl CostMetric for RootMeanSquaredError {
    fn evaluate(&self, residuals: &[f64]) -> f64 {
        if residuals.is_empty() {
            return 0.0;
        }

        let n = residuals.len() as f64;
        let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / n;
        mse.sqrt()
    }

    fn name(&self) -> &'static str {
        "rmse"
    }

    fn evaluate_with_sensitivities(
        &self,
        residuals: &[f64],
        sensitivities: &[DMatrix<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        Some(self.evaluate_with_sensitivities_impl(residuals, sensitivities))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GaussianNll {
    variance: f64,
    log_term: f64, // Cache the constant term
}

impl GaussianNll {
    /// Creates a new Gaussian negative log-likelihood cost metric.
    ///
    /// # Arguments
    /// * `variance` - The variance of the Gaussian distribution (must be positive)
    ///
    /// # Panics
    /// Panics if variance is not positive and finite
    pub fn new(variance: f64) -> Self {
        assert!(
            variance > 0.0 && variance.is_finite(),
            "Variance must be positive and finite, got {}",
            variance
        );

        let log_term = (2.0 * PI * variance).ln();
        Self { variance, log_term }
    }

    /// Creates a new Gaussian NLL with variance clamped to a valid range.
    /// Use this when you want to handle invalid inputs gracefully.
    pub fn new_clamped(variance: f64) -> Self {
        let clamped = variance.clamp(f64::EPSILON, f64::MAX);
        Self::new(clamped)
    }
}

impl Default for GaussianNll {
    fn default() -> Self {
        Self::new(1.0)
    }
}
impl GaussianNll {
    fn evaluate_with_sensitivities_impl(
        &self,
        residuals: &[f64],
        sensitivities: &[DMatrix<f64>],
    ) -> (f64, Vec<f64>) {
        if residuals.is_empty() {
            return (0.0, Vec::new());
        }

        let n = residuals.len() as f64;
        let sse: f64 = residuals.iter().map(|&r| r * r).sum();

        let cost = 0.5 * n * self.log_term + 0.5 * sse / self.variance;

        if sensitivities.is_empty() {
            return (cost, Vec::new());
        }

        let mut grad_nll = vec![0.0; sensitivities.len()];

        for (param_idx, sens) in sensitivities.iter().enumerate() {
            assert_eq!(
                sens.len(),
                residuals.len(),
                "sensitivity matrix must have the same number of elements as residuals",
            );

            let dot: f64 = sens.iter().zip(residuals.iter()).map(|(s, r)| r * s).sum();
            grad_nll[param_idx] = dot;
        }
        // Gradient: d(GaussianNll)/dp = d(GaussianNll)/dy * dy/dp
        // Gradient: d(GaussianNll)/dp = residual/variance * sensitivities
        let gradient: Vec<f64> = grad_nll.into_iter().map(|g| g / self.variance).collect();

        (cost, gradient)
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
        0.5 * n * self.log_term + 0.5 * sse / self.variance
    }

    fn name(&self) -> &'static str {
        "gaussian_nll"
    }

    fn evaluate_with_sensitivities(
        &self,
        residuals: &[f64],
        sensitivities: &[DMatrix<f64>],
    ) -> Option<(f64, Vec<f64>)> {
        Some(self.evaluate_with_sensitivities_impl(residuals, sensitivities))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_basic() {
        let metric = SumSquaredError;
        let residuals = vec![1.0, 2.0, 3.0];
        assert_eq!(metric.evaluate(&residuals), 14.0);
    }

    #[test]
    fn test_sse_with_gradient() {
        let metric = SumSquaredError;
        let residuals = vec![1.0, 2.0];
        let sensitivities = vec![0.5, 0.5];
        let sens_matrix = DMatrix::from_vec(2, 1, sensitivities);
        let (cost, grad) = metric
            .evaluate_with_sensitivities(&residuals, &[sens_matrix])
            .expect("SumSquaredError should support gradient evaluation");
        assert_eq!(cost, 5.0);
        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0], 3.0); // 2*1*0.5 + 2*2*0.5
    }

    #[test]
    fn test_rmse_basic() {
        let metric = RootMeanSquaredError;
        let residuals = vec![1.0, 2.0, 3.0];
        let expected = (14.0 / 3.0_f64).sqrt();
        assert!((metric.evaluate(&residuals) - expected).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Variance must be positive")]
    fn test_gaussian_nll_invalid_variance() {
        GaussianNll::new(-1.0);
    }

    #[test]
    fn test_gaussian_nll_clamped() {
        let metric = GaussianNll::new_clamped(-1.0);
        assert!(metric.variance > 0.0);
    }
}
