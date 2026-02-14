use rand::prelude::StdRng;
use rand::Rng;
use rand_distr::StandardNormal;
use std::fmt;
use std::ops::RangeInclusive;
use std::sync::Arc;

/// Type alias for a point in parameter space
pub type Point = Vec<f64>;

/// Result of calling `ask()` on an optimiser or sampler state
///
/// This generic enum allows both optimisers and samplers to use the same
/// ask/tell interface while returning their own result types.
///
/// # Type Parameters
/// - `R`: The result type (e.g., `OptimisationResults` or `SamplingResults`)
#[derive(Clone, Debug)]
pub enum AskResult<R> {
    /// Evaluate these points and call `tell()` with the results
    Evaluate(Arc<[Point]>),
    /// Algorithm has finished - contains final results
    Done(R),
}

#[derive(Debug, Clone)]
pub struct Unbounded;

/// Represents parameter bounds for optimisation and sampling algorithms.
///
/// Each dimension has a lower and upper bound represented as a `RangeInclusive<f64>`.
/// Bounds can be finite (e.g., `[0.0, 1.0]`) or infinite (e.g., `[-∞, ∞]`).
///
/// # Examples
///
/// ```
/// use diffid::common::Bounds;
///
/// // Create bounds for a two-dimensional parameter-space
/// let bounds = Bounds::new(vec![(0.0, 1.0), (-5.0, 5.0)]);
/// assert_eq!(bounds.dimension(), 2);
///
/// // Using From trait
/// let bounds: Bounds = vec![(0.0, 1.0), (-5.0, 5.0)].into();
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bounds {
    pub(crate) limits: Vec<RangeInclusive<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundsDimensionError {
    expected: usize,
    got: usize,
}

impl BoundsDimensionError {
    pub fn expected(&self) -> usize {
        self.expected
    }

    pub fn got(&self) -> usize {
        self.got
    }
}

impl fmt::Display for BoundsDimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bounds dimension mismatch: expected {}, got {}",
            self.expected, self.got
        )
    }
}

impl std::error::Error for BoundsDimensionError {}

impl Bounds {
    /// Creates a new `Bounds` from a vector of (lower, upper) tuples.
    ///
    /// # Arguments
    ///
    /// * `limits` - Vector of (lower, upper) bound pairs for each dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use diffid::common::Bounds;
    ///
    /// let bounds = Bounds::new(vec![(0.0, 1.0), (-10.0, 10.0)]);
    /// ```
    pub fn new(limits: Vec<(f64, f64)>) -> Self {
        limits.into()
    }

    /// Returns the parameter-space dimensionality of the bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use diffid::common::Bounds;
    ///
    /// let bounds = Bounds::new(vec![(0.0, 1.0), (-5.0, 5.0), (0.0, 100.0)]);
    /// assert_eq!(bounds.dimension(), 3);
    /// ```
    pub fn dimension(&self) -> usize {
        self.limits.len()
    }

    /// Creates unbounded ranges for n-dimensions
    pub fn unbounded(dimensions: usize) -> Self {
        Self {
            limits: vec![f64::NEG_INFINITY..=f64::INFINITY; dimensions],
        }
    }

    /// Creates unbounded ranges from the dimensions of a point
    ///
    /// # Examples
    /// ```
    /// use diffid::common::Bounds;
    ///
    /// let initial = [0.0];
    /// let bounds = Bounds::unbounded_like(&initial);
    /// assert_eq!(bounds.dimension(), 1)
    /// ```
    pub fn unbounded_like<T: AsRef<[f64]>>(point: T) -> Self {
        Self::unbounded(point.as_ref().len())
    }

    /// Returns True if all ranges are unbounded
    pub fn is_unbounded(&self) -> bool {
        self.limits
            .iter()
            .all(|r| r.start().is_infinite() && r.end().is_infinite())
    }

    /// Returns a reference to the inner limits as a slice
    pub fn limits(&self) -> &[RangeInclusive<f64>] {
        &self.limits
    }

    /// Validate that bounds dimensionality matches an expected parameter dimension.
    ///
    /// # Errors
    ///
    /// Returns [`BoundsDimensionError`] when `expected` differs from `self.dimension()`.
    pub fn validate_dimension(&self, expected: usize) -> Result<(), BoundsDimensionError> {
        let got = self.dimension();
        if got == expected {
            Ok(())
        } else {
            Err(BoundsDimensionError { expected, got })
        }
    }

    /// Clamps a vector of positions in-place to remain within the bounds.
    ///
    /// Each element is clamped to its corresponding bound range. For infinite bounds,
    /// values are clamped to the finite bound (if any) or left unchanged if both are infinite.
    ///
    /// # Arguments
    ///
    /// * `positions` - Mutable slice of position values to clamp
    ///
    /// # Examples
    ///
    /// ```
    /// use diffid::common::Bounds;
    ///
    /// let bounds = Bounds::new(vec![(0.0, 1.0), (-5.0, 5.0)]);
    /// let mut pos = vec![1.5, -10.0];
    /// bounds.clamp(&mut pos);
    /// assert_eq!(pos, vec![1.0, -5.0]);
    /// ```
    pub fn clamp(&self, positions: &mut [f64]) {
        positions
            .iter_mut()
            .zip(&self.limits)
            .for_each(|(value, range)| {
                *value = value.clamp(*range.start(), *range.end());
            });
    }

    /// Draws a random position inside the bounds with optional Gaussian expansion.
    ///
    /// For each dimension:
    /// - **Finite bounds**: Samples uniformly within the range, then adds Gaussian noise
    ///   with `sigma = width * expansion_factor`. The result is clamped to the bounds.
    /// - **Infinite bounds**: Samples from a Gaussian centered at the finite bound (or 0.0
    ///   if both are infinite) with `sigma = expansion_factor`.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `expansion_factor` - Controls the scale of Gaussian noise. Choose a value
    ///   appropriate for your problem scale. Common values: 0.01-0.1 for exploration.
    ///
    /// # Examples
    ///
    /// ```
    /// use diffid::common::Bounds;
    /// use rand::SeedableRng;
    /// use rand::prelude::StdRng;
    ///
    /// let bounds = Bounds::new(vec![(0.0, 1.0), (-5.0, 5.0)]);
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let position = bounds.sample(&mut rng, 0.05);
    /// assert_eq!(position.len(), 2);
    /// ```
    pub fn sample(&self, rng: &mut StdRng, expansion_factor: f64) -> Vec<f64> {
        let scale = expansion_factor.abs();

        self.limits
            .iter()
            .map(|range| {
                let lo = *range.start();
                let hi = *range.end();

                if lo.is_finite() && hi.is_finite() {
                    // finite bounds
                    let base = rng.random_range(lo..=hi);
                    let width = hi - lo;
                    let sigma = width * scale;
                    let draw = rng.sample::<f64, _>(StandardNormal);
                    draw.mul_add(sigma, base).clamp(lo, hi)
                } else {
                    // At least one bound is non-finite
                    let base = if lo.is_finite() {
                        lo
                    } else if hi.is_finite() {
                        hi
                    } else {
                        0.0
                    };
                    let sigma = scale;
                    let offset = rng.sample::<f64, _>(StandardNormal);
                    offset.mul_add(sigma, base)
                }
            })
            .collect()
    }
}

impl From<Vec<(f64, f64)>> for Bounds {
    /// Converts a vector of (lower, upper) tuples into `Bounds`.
    ///
    /// # Examples
    ///
    /// ```
    /// use diffid::common::Bounds;
    ///
    /// let bounds: Bounds = vec![(0.0, 1.0), (-5.0, 5.0)].into();
    /// assert_eq!(bounds.dimension(), 2);
    /// ```
    fn from(tuples: Vec<(f64, f64)>) -> Self {
        Self {
            limits: tuples.into_iter().map(|(lo, hi)| lo..=hi).collect(),
        }
    }
}
