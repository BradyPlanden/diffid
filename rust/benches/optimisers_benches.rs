use chronopt::common::Bounds;
use chronopt::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn bench_nelder_mead_quadratic(c: &mut Criterion) {
    let problem = ScalarProblemBuilder::new()
        .with_function(|x: &[f64]| {
            let x0 = x[0] - 1.5;
            let x1 = x[1] + 0.5;
            x0 * x0 + x1 * x1
        })
        .build()
        .expect("failed to build quadratic problem");
    let optimiser = NelderMead::new()
        .with_max_iter(200)
        .with_threshold(1e-8)
        .with_step_size(0.6)
        .with_position_tolerance(1e-6);
    let initial = vec![5.0_f64, -4.0_f64];

    c.bench_function("nelder_mead_quadratic", move |b| {
        let problem = &problem;
        let optimiser = optimiser.clone();
        let initial = initial.clone();
        b.iter(|| {
            let result = optimiser.run(
                |x| problem.evaluate(x),
                black_box(initial.clone()),
                Bounds::unbounded(2),
            );
            black_box(result.value);
        });
    });
}

fn bench_cmaes_quadratic(c: &mut Criterion) {
    let problem = ScalarProblemBuilder::new()
        .with_function(|x: &[f64]| {
            let x0 = x[0] - 1.5;
            let x1 = x[1] + 0.5;
            x0 * x0 + x1 * x1
        })
        .build()
        .expect("failed to build quadratic problem");
    let optimiser = CMAES::new()
        .with_max_iter(200)
        .with_threshold(1e-8)
        .with_step_size(0.6)
        .with_seed(42);
    let initial = vec![5.0_f64, -4.0_f64];

    c.bench_function("cmaes_quadratic", move |b| {
        let problem = &problem;
        let optimiser = optimiser.clone();
        let initial = initial.clone();
        b.iter(|| {
            let result = optimiser.run(
                |x| problem.evaluate(x),
                black_box(initial.clone()),
                Bounds::unbounded(2),
            );
            black_box(result.value);
        });
    });
}

fn bench_adam_quadratic(c: &mut Criterion) {
    let problem = ScalarProblemBuilder::new()
        .with_function(|x: &[f64]| {
            let x0 = x[0] - 1.5;
            let x1 = x[1] + 0.5;
            x0 * x0 + x1 * x1
        })
        .with_gradient(|x: &[f64]| vec![2.0 * (x[0] - 1.5), 2.0 * (x[1] + 0.5)])
        .build()
        .expect("failed to build quadratic problem with gradient");
    let optimiser = Adam::new()
        .with_step_size(0.1)
        .with_max_iter(200)
        .with_threshold(1e-8);
    let initial = vec![5.0_f64, -4.0_f64];

    c.bench_function("adam_quadratic", move |b| {
        let problem = &problem;
        let optimiser = optimiser.clone();
        let initial = initial.clone();
        b.iter(|| {
            let result = optimiser.run(
                |x| {
                    let (val, grad_opt) = problem
                        .evaluate_with_gradient(x)
                        .expect("evaluate_with_gradient failed");
                    (val, grad_opt.expect("gradient should be available"))
                },
                black_box(initial.clone()),
                Bounds::unbounded(2),
            );
            black_box(result.value);
        });
    });
}

fn optimiser_benches(c: &mut Criterion) {
    bench_nelder_mead_quadratic(c);
    bench_cmaes_quadratic(c);
    bench_adam_quadratic(c);
}

criterion_group!(name = optimisers; config = Criterion::default().measurement_time(Duration::from_secs(10)); targets = optimiser_benches);
criterion_main!(optimisers);
