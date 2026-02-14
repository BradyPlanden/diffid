# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Features
- Optimiser evaluation-failure semantics aligned across Adam, Nelder-Mead, CMA-ES, and Diffsol objective handling (strict fail with `FunctionEvaluationFailed`).
- Added allocation-light single-point ask path for Adam and Nelder-Mead while preserving backward-compatible batch-shaped Ask/Tell APIs.
- Added Adam history toggle (`with_history`) with cheap-by-default behavior.
- Decoupled batch-evaluation routing from parallel-evaluation capability so optimise/sample can use batch paths whenever objectives support batching.

### Performance
- Reduced hot-loop allocation pressure in Adam, CMA-ES, and vector objective evaluation paths.

### Documentation
- Updated optimiser docs/guides to describe strict failure behavior, bounds dimension validation, Adam history capture, and batch routing behavior.

## [0.3.0] - 2026-01-25

### Features
- Python package renamed from `chronopt` to `diffid`
- Ask-Tell interface for optimisers and samplers, enabling stateful step-by-step execution
- Optimisers unified into an `Optimiser` enum with shared `run()` API
- `ParameterRange` type with improved Bounds API supporting unbounded parameters
- Python bindings enhanced with magic methods (`__repr__`, `__str__`, etc.) for better REPL experience
- Cached Diffsol parallelisation with single solver build per thread and improved multithreaded error management
- Improved error hierarchy with `EvaluationError`, `TellError`, and `ProblemBuilderError` distinctions
- Added MCMC proposals for Dynamic Nested Sampling with corresponding benchmarks

### Breaking Changes
- Result attributes renamed: `sigma0` → `step_size`, standardised to `evaluations` and `iterations`
- Codebase aligned to British English spelling conventions
- Samplers refactored with Ask-Tell interface, matching optimiser patterns
- Module restructure: `types.rs` moved to library level, error types consolidated
- `ScalarEvaluation` and `GradientEvaluation` now use `TryFrom` trait implementations
- `Objective` trait extended with `has_gradient()` method
- Optimiser `.tell()` and `.run()` methods use relaxed error management

### Fixes
- Incorrectly indexed proposal storage in Dynamic Nested Sampling
- Patience convergence criterion now uses proper type conversions

## [0.2.0] - 2025-12-01

### Features
- Dynamic Nested Sampling (DNS) sampler with Rust and Python bindings, including a dedicated scheduler, proposal generator, result struct, and corresponding tests and stubs.
- Support for sensitivities in `DiffsolProblem` across dense and sparse backends, exposing gradient support through `CostMetric` and `problem.evaluate_with_grad(...)`.
- Problem-level parallel evaluation control, parallel execution support for DNS, and a `time` attribute on sampler results, with parallel sampler tests.
- Adds ADAM optimisation algorithm with Python bindings and stub support.
- Multi-cost / vectored cost support, including cost weighting utilities for composing and weighting multiple metrics.
- Diffsol backend configuration updated to use LLVM-18 for sensitivity calculations and improved performance.
- Adds benchmarking suite for Diffsol, Optimiser, and Samplers via `cargo bench`

### Fixes
- Refactor DiffsolProblem's evaluate function for reduced clone's and aligned error management
- Several robustness fixes to Diffsol integration, including catching panics during failure cases and rebuilding with penalties.
- CMA-ES bug fixes and bound handling improvements, with associated tests.
- Corrected `PyProblem.evaluate` gradient argument handling on the Python side.
- Miscellaneous clippy warnings and CI-related issues.

### Breaking
- Vector-valued optimisation via `VectorProblemBuilder` and associated `Problem` type, alongside renamed `ScalarProblemBuilder` for scalar objectives.
- `build` methods on problem builders no longer consume the builder, enabling repeated `build` calls and multi-build workflows.

### Examples
- New dynamic nested sampling and model evidence examples with updated evidence values using more reasonable live point counts.
- New bicycle model identification example with model selection via evidence.
- Expanded predator–prey example suite, including a dedicated subdirectory comparing solver backends and a Diffeqpy-based identification workflow.
- Updated JAX predator–prey example to use Diffrax.

## [0.1.0] - 2025-11-05

### Added
- Initial Chronopt release combining the Rust optimisation core with Python bindings via PyO3 and maturin packaging.
- Diffsol integration with configurable dense and sparse backends, gradient support, and builder APIs for solver configuration.
- Suite of optimisation algorithms including Nelder–Mead, CMA-ES, patience-based halting, and parallel evaluation helpers.
- Cost metric implementations (SSE, RMSE, NLL) with Python-facing builder helpers and validation.
- Sampling module with the initial Metropolis–Hastings implementation and worked Python examples (e.g., bouncy ball, predator–prey, Lotka–Volterra).
- Extensive unit and integration tests covering builders, diffsol backends, sampling, and mathematical benchmarking suites.
- Automated stub generation helpers, documentation updates, and GitHub Actions workflows for CI and trusted-publisher PyPI releases.
