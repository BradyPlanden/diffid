# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-11-05

### Added
- Initial Chronopt release combining the Rust optimisation core with Python bindings via PyO3 and maturin packaging.
- Diffsol integration with configurable dense and sparse backends, gradient support, and builder APIs for solver configuration.
- Suite of optimisation algorithms including Nelder–Mead, CMA-ES, patience-based halting, and parallel evaluation helpers.
- Cost metric implementations (SSE, RMSE, NLL) with Python-facing builder helpers and validation.
- Sampling module with the initial Metropolis–Hastings implementation and worked Python examples (e.g., bouncy ball, predator–prey, Lotka–Volterra).
- Extensive unit and integration tests covering builders, diffsol backends, sampling, and mathematical benchmarking suites.
- Automated stub generation helpers, documentation updates, and GitHub Actions workflows for CI and trusted-publisher PyPI releases.
