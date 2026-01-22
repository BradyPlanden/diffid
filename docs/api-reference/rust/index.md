# Rust API Documentation

Chronopt's Rust core provides high-performance implementations of all optimisation and sampling algorithms.

## Official Documentation

The complete Rust API documentation is hosted on docs.rs:

**[:octicons-arrow-right-24: Chronopt Rust Documentation on docs.rs](https://docs.rs/chronopt/latest/chronopt/)**

## When to Use the Rust API

Consider using the Rust crate directly when:

- **Maximum performance** is critical
- Building **Rust-native applications**
- Need **zero-copy** data handling
- Deploying to **embedded systems** or **constrained environments**
- Building **custom tooling** around Chronopt

For most users, the Python API provides excellent performance with easier integration.

## Crate Structure

```
chronopt/
├── builders/           # Problem builders (ScalarBuilder, DiffsolBuilder, etc.)
├── optimisers/         # Optimisation algorithms
│   ├── nelder_mead/    # Nelder-Mead simplex
│   ├── cmaes/          # CMA-ES evolution strategy
│   └── adam/           # Adam gradient descent
├── sampler/            # MCMC and nested sampling
├── cost/               # Cost metrics (SSE, RMSE, GaussianNLL)
├── problem/            # Problem types and evaluation
└── common/             # Shared types and utilities
```

## Quick Example

```rust
use chronopt::prelude::*;
use ndarray::array;

// Define objective function
fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
}

fn main() {
    // Build problem
    let builder = ScalarBuilder::new()
        .with_objective(rosenbrock)
        .with_parameter("x", 1.5)
        .with_parameter("y", -1.5);

    let problem = builder.build();

    // Run optimisation
    let result = problem.optimise();

    println!("Optimal parameters: {:?}", result.x);
    println!("Objective value: {:.3e}", result.value);
    println!("Success: {}", result.success);
}
```

## Adding Chronopt to Your Project

Add to your `Cargo.toml`:

```toml
[dependencies]
chronopt = "0.2"
ndarray = "0.15"
```

For ODE support with DiffSL:

```toml
[dependencies]
chronopt = { version = "0.2", features = ["diffsol"] }
```

## Key Rust Features

### Zero-Copy Performance

The Rust API avoids unnecessary allocations and copies:

```rust
// Efficient in-place evaluation
let mut output = vec![0.0; n];
problem.evaluate_into(&params, &mut output)?;
```

### Type Safety

Strong typing catches errors at compile time:

```rust
// Compiler ensures correct types
let builder: ScalarBuilder = ScalarBuilder::new()
    .with_objective(objective)
    .with_parameter("x", 1.0);

let problem: Problem = builder.build();
```

### Parallel Execution

Native Rayon parallelism:

```rust
use rayon::prelude::*;

let results: Vec<_> = initial_guesses
    .par_iter()
    .map(|guess| optimiser.run(&problem, guess))
    .collect();
```

## Documentation Generation

Generate local documentation:

```bash
cd rust
cargo doc --open --no-deps
```

This builds and opens the full API documentation in your browser.

## Contributing to Rust Core

See the [Contributing Guide](../../development/contributing.md) and [Architecture](../../development/architecture.md) docs for:

- Code organization and patterns
- Adding new optimisers or samplers
- Implementing custom cost metrics
- Testing strategies
- PyO3 binding guidelines

## Module Documentation

Key modules (click through on docs.rs for full details):

### `builders`

Problem construction with fluent API.

```rust
pub use chronopt::builders::{ScalarBuilder, DiffsolBuilder, VectorBuilder};
```

### `optimisers`

Optimisation algorithms.

```rust
pub use chronopt::optimisers::{NelderMead, CMAES, Adam};
```

### `cost`

Cost metrics for objective functions.

```rust
pub use chronopt::cost::{CostMetric, SSE, RMSE, GaussianNLL};
```

### `problem`

Problem types and evaluation.

```rust
pub use chronopt::problem::{Problem, ScalarProblem, VectorProblem};
```

## Performance Tips

1. **Use release builds**: `cargo build --release` (10-100x faster than debug)
2. **Profile first**: Use `cargo flamegraph` to identify bottlenecks
3. **Parallel backends**: Enable Rayon for CMA-ES and Diffsol
4. **Sparse matrices**: Use sparse backend for large ODE systems
5. **Avoid allocations**: Reuse buffers in hot loops

## Examples

The repository includes Rust examples:

```bash
cd rust
cargo run --example rosenbrock
cargo run --example ode_fitting
```

Browse examples on GitHub: [rust/examples/](https://github.com/bradyplanden/chronopt/tree/main/rust/examples)

## See Also

- [Python API Reference](../index.md)
- [Architecture](../../development/architecture.md)
- [Contributing](../../development/contributing.md)
