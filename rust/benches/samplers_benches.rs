use criterion::{black_box, criterion_group, criterion_main, Criterion};
use diffid::prelude::*;
use std::time::Duration;

fn bench_metropolis_hastings_gaussian(c: &mut Criterion) {
    let problem = ScalarProblemBuilder::new()
        .with_function(|x: &[f64]| {
            let diff = x[0] - 0.5;
            0.5 * diff * diff
        })
        .with_parameter("x", 0.6, (-5.0, 5.0))
        .build()
        .expect("failed to build gaussian problem");
    let sampler = MetropolisHastings::new()
        .with_num_chains(4)
        .with_iterations(500)
        .with_step_size(0.3)
        .with_seed(42);
    let initial = vec![0.5_f64];

    c.bench_function("metropolis_hastings_gaussian", move |b| {
        let problem = &problem;
        let sampler = sampler.clone();
        let initial = initial.clone();
        b.iter(|| {
            let samples = sampler.run(
                |x| problem.evaluate(x),
                black_box(initial.clone()),
                Bounds::new(vec![(-5.0, 5.0)]),
            );
            black_box(samples.draws());
        });
    });
}

fn bench_dynamic_nested_gaussian(c: &mut Criterion) {
    let problem = ScalarProblemBuilder::new()
        .with_function(|x: &[f64]| {
            let diff = x[0] - 0.5;
            0.5 * diff * diff
        })
        .with_parameter("x", 0.6, (-5.0, 5.0))
        .build()
        .expect("failed to build gaussian problem");
    let sampler = DynamicNestedSampler::new()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_termination_tolerance(1e-3)
        .with_seed(37);
    let initial = vec![0.5_f64];

    c.bench_function("dynamic_nested_gaussian", move |b| {
        let problem = &problem;
        let sampler = sampler.clone();
        let initial = initial.clone();
        b.iter(|| {
            let nested = sampler.run(
                |x| problem.evaluate(x),
                black_box(initial.clone()),
                Bounds::new(vec![(-5.0, 5.0)]),
            );
            black_box((nested.draws(), nested.log_evidence()));
        });
    });
}

fn sampler_benches(c: &mut Criterion) {
    bench_metropolis_hastings_gaussian(c);
    bench_dynamic_nested_gaussian(c);
}

criterion_group!(name = samplers; config = Criterion::default().measurement_time(Duration::from_secs(10)); targets = sampler_benches);
criterion_main!(samplers);
