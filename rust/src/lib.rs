pub mod optimisers;
pub mod problem;

#[cfg(test)]
mod tests {

    use crate::optimisers::NelderMead;
    use crate::problem::Builder;

    #[test]
    fn test_simple_optimisation() {
        let problem = Builder::new()
            .with_objective(|x: &[f64]| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();

        let optimiser = NelderMead::new().with_max_iter(500).with_sigma0(0.4);
        let result = optimiser.run(&problem, vec![1.0, 1.0]);

        assert!(result.success);
        assert!(
            result.fun < 0.01,
            "Expected fun < 0.01, but got: {}",
            result.fun
        );
    }
}
