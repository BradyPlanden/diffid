use crate::problem::Problem;

// Core behaviour shared by all optimisers
pub trait Optimiser {
    fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults;
}

// Optimiser traits
pub trait WithMaxIter: Sized {
    fn set_max_iter(&mut self, max_iter: usize);
    fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.set_max_iter(max_iter);
        self
    }
}

pub trait WithThreshold: Sized {
    fn set_threshold(&mut self, threshold: f64);
    fn with_threshold(mut self, threshold: f64) -> Self {
        self.set_threshold(threshold);
        self
    }
}

pub trait WithSigma0: Sized {
    fn set_sigma0(&mut self, sigma0: f64);
    fn with_sigma0(mut self, sigma0: f64) -> Self {
        self.set_sigma0(sigma0);
        self
    }
}

// Nelder-Mead optimiser
pub struct NelderMead {
    max_iter: usize,
    threshold: f64,
    sigma0: f64,
}

impl NelderMead {
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            threshold: 1e-6,
            sigma0: 0.1,
        }
    }

    pub fn run(&self, problem: &Problem, initial: Vec<f64>) -> OptimisationResults {
        // Simplified Nelder-Mead for testing
        let mut x = initial;
        let mut best_val = problem.evaluate(&x);
        let mut iterations = 0;

        for i in 0..self.max_iter {
            iterations += 1;

            // Simplified step
            let perturbation = self.sigma0 / (i as f64 + 1.0);
            let mut improved = false;

            for j in 0..x.len() {
                for &sign in &[-1.0, 1.0] {
                    let mut x_new = x.clone();
                    x_new[j] += sign * perturbation;

                    let val = problem.evaluate(&x_new);
                    if val < best_val {
                        x = x_new;
                        best_val = val;
                        improved = true;
                    }
                }
            }

            if !improved && perturbation < self.threshold {
                break;
            }
        }
        OptimisationResults {
            x,
            fun: best_val,
            nit: iterations,
            success: true,
        }
    }
}

impl WithMaxIter for NelderMead {
    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }
}

impl WithThreshold for NelderMead {
    fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}

impl WithSigma0 for NelderMead {
    fn set_sigma0(&mut self, sigma0: f64) {
        self.sigma0 = sigma0;
    }
}

impl Default for NelderMead {
    fn default() -> Self {
        Self::new()
    }
}

// Results object
#[derive(Debug, Clone)]
pub struct OptimisationResults {
    pub x: Vec<f64>,
    pub fun: f64,
    pub nit: usize,
    pub success: bool,
}

impl OptimisationResults {
    fn __repr__(&self) -> String {
        format!(
            "OptimizationResults(x={:?}, fun={:.6}, nit={}, success={}))",
            self.x, self.fun, self.nit, self.success
        )
    }
}
