use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use diffid::builders::DiffsolBackend;
use diffid::prelude::*;
use nalgebra::DMatrix;
use std::time::Duration;

macro_rules! build_logistic_problem {
    ($backend:expr, $parallel:expr) => {{
        let dsl = r#"
in_i { r = 1, k = 1 }
r { 1 }
k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"#;

        let t_span: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let data_values: Vec<f64> = t_span.iter().map(|t| 0.1 * (*t).exp()).collect();
        let data = DMatrix::from_fn(t_span.len(), 2, |i, j| match j {
            0 => t_span[i],
            1 => data_values[i],
            _ => unreachable!(),
        });

        DiffsolProblemBuilder::new()
            .with_diffsl(dsl.to_string())
            .with_data(data)
            .with_parameter("r", 1.0, (0.1, 3.0))
            .with_parameter("k", 1.0, (0.5, 2.0))
            .with_backend($backend)
            .with_parallel($parallel)
            .build()
            .expect("failed to build diffsol problem for benchmarks")
    }};
}

fn bench_diffsol_single_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffsol_single_eval");
    let initial = vec![1.0_f64, 1.0_f64];

    for backend in [DiffsolBackend::Dense, DiffsolBackend::Sparse] {
        let problem = build_logistic_problem!(backend, false);
        group.bench_with_input(
            BenchmarkId::new("evaluate", format!("{:?}", backend)),
            &problem,
            |b, problem| {
                b.iter(|| {
                    let cost = problem
                        .evaluate(black_box(&initial))
                        .expect("evaluation failed");
                    black_box(cost);
                });
            },
        );
    }

    group.finish();
}

fn bench_diffsol_population_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffsol_population_eval");

    let population: Vec<Vec<f64>> = (0..64)
        .map(|i| {
            let scale = 0.8 + (i as f64) * 0.01;
            vec![1.0 * scale, 1.0 / scale]
        })
        .collect();

    for backend in [DiffsolBackend::Dense, DiffsolBackend::Sparse] {
        for &parallel in &[false, true] {
            let label = format!("{:?}_parallel={}", backend, parallel);
            let problem = build_logistic_problem!(backend, parallel);

            group.bench_with_input(
                BenchmarkId::new("evaluate_population", label),
                &problem,
                |b, problem| {
                    b.iter(|| {
                        let results = problem.evaluate_population(black_box(&population));
                        for res in results {
                            let _ = black_box(res.expect("population evaluation failed"));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn diffsol_benches(c: &mut Criterion) {
    bench_diffsol_single_eval(c);
    bench_diffsol_population_eval(c);
}

criterion_group!(name = diffsol; config = Criterion::default().measurement_time(Duration::from_secs(10)); targets = diffsol_benches);
criterion_main!(diffsol);
