## Chronopt API Implementation Plan (aligned to current code and examples)

### Current capabilities (as of now)
- **Rust core**:
  - `problem::Builder` builds a `Problem` from a callable objective and optional numeric config.
  - `Problem::evaluate(&[f64]) -> f64` and config accessors exist; no `optimize` method yet.
  - `optimisers::NelderMead` with `with_max_iter`, `with_threshold`, `with_sigma0`, and `run(problem, x0)` returning `OptimisationResults { x, fun, nit, success }`.
- **Python bindings**:
  - Exposed classes: `Builder`, `Problem`, `NelderMead`, `OptimisationResults`.
  - `Builder.add_callable(callable)` and `Builder.with_config(key, value)`; `Builder.build()` creates a `Problem`.
  - `NelderMead.with_max_iter(...)` and `.run(problem, x0)`; `OptimisationResults` exposes getters.
- **Examples status**:
  - `examples/python_problem.py` is compatible today (uses `Builder` + `NelderMead.run`).
  - `examples/pure_python_problem.py` expects `PythonBuilder`, `.add_parameter`, and `Problem.optimize(...)` — not implemented yet.
  - `examples/diffsol_problem.py` expects `builder.Diffsol` and `Hamiltonian` — not implemented yet.
  - `examples/model_evidence.py` expects `DynamicNestedSampler` — not implemented yet.

### Target Python API (to satisfy examples)
- **Callable problems**: `chron.PythonBuilder().add_callable(fn).add_parameter("x").set_optimiser(chron.NelderMead().with_max_iter(...)).build()`
- **Simpler builder alias**: `chron.Builder()` remains supported (alias of `PythonBuilder`) for backward compatibility.
- **Problem.optimize**: `problem.optimize(x0: Optional[List[float]]=None, optimiser: Optional[NelderMead]=None)` returning `OptimisationResults` with `.x`, `.fun`, etc.
- **Diffsol builder**: `chron.builder.Diffsol().add_diffsl(ds).add_data(data).add_config(config).add_params(params).build()`
- **Sampler (HMC)**: `chron.Hamiltonian().set_number_of_chains(...).set_parallel(True).run(problem, x0=...)` returning `samples.mean_x`.
- **Evidence (optional MVP)**: `chron.DynamicNestedSampler(problem).run()` returning `log_z` (stub acceptable initially).

### Phase 1 — Close the gap for callable optimisation
- **Rust core**
  - Add `Problem::optimize(initial: Option<Vec<f64>>, optimiser: Option<&dyn Optimiser>) -> OptimisationResults`.
  - Default optimiser: `NelderMead::new()` when `optimiser` is `None`; require `initial` or infer zeros of length `dimension()` (see below).
  - Add `Problem::dimension() -> usize` for sanity and default initialisation.
- **Python bindings**
  - Expose `Problem.optimize(initial: Optional[List[float]] = None, optimiser: Optional[NelderMead] = None)`.
  - Add a readable `__repr__` for `OptimisationResults`.
  - Add `PythonBuilder` as an alias to `Builder` and implement `.add_parameter(name: str, prior: Optional[Any] = None)` to record parameter order and optional prior (affects `dimension()` only for now).
  - Support `Builder.set_optimiser(opt)` as sugar to store a default optimiser used by `Problem.optimize()` when `optimiser=None`.

### Phase 2 — Hamiltonian sampler (MVP)
- **Rust core**
  - New module `samplers::hamiltonian` with struct `Hamiltonian { num_chains, parallel, num_steps, step_size }`.
  - Implement simple HMC with leapfrog, diagonal mass, fixed step size.
  - Parallelize chains with `rayon`.
  - Output `Samples { chains: Vec<Vec<Vec<f64>>>, mean_x: Vec<f64>, draws: usize }`.
- **Python bindings**
  - Class `Hamiltonian` with:
    - `set_number_of_chains(int)`
    - `set_parallel(bool)`
    - `with_num_steps(int)`, `with_step_size(float)` (optional for MVP)
    - `run(problem: Problem, x0: List[float]) -> Samples`
  - Treat objective as negative log-density for MVP.

### Phase 3 — Diffsol builder and simulation representation
- **Core problem abstraction**
  - Extend `Problem` to support multiple kinds via `enum ProblemKind { Callable(ObjectiveFn), Diffsol(DiffsolSim) }`.
  - `Problem::evaluate(x)` dispatches by kind.
  - Add a `Cost` implementation, use sum-of-squares between simulated trajectory and `data`.
- **DiffsolSim (MVP)**
  - Fields: parsed DSL, `params`, `config`.
  - Hook into diffsol ODE builder, `build_from_diffsl` (start with minimal grammar required by the example).
  - Solver: BDF or RK with stable defaults.
- **Python builder**
  - `chron.builder.Diffsol` with:
    - `.add_diffsl(str)`
    - `.add_data(list | np.ndarray)`
    - `.add_config(dict)`
    - `.add_params(dict)`
    - `.build() -> Problem`

### Phase 4 — Evidence (optional) and ergonomics
- **Dynamic nested sampler (stub)**
  - `chron.DynamicNestedSampler(problem).run() -> float` returning a placeholder evidence value for now; wire real implementation later or use a simple importance sampling baseline.
- **Config management**
  - Keep `HashMap<String, f64>` for numeric config.
  - Typed getters for common keys: `rtol`, `atol`, `step_size`, `num_steps`.
  - Python: accept dicts of floats; validate known keys.
- **Initial value inference**
  - Preserve parameter insertion order from Python to construct `x0`.
  - Expose `Problem.dimension()` for sanity checks.

### Phase 5 — Testing, examples, and docs
- Extend tests:
  - `Problem.optimize()` matches `NelderMead.run(problem, x0)`.
  - `Hamiltonian` returns `mean_x` near MAP on convex objective.
  - Diffsol builder produces a problem that optimizes the logistic example.
  - `DynamicNestedSampler.run()` returns a finite value on a toy problem.
- Examples:
  - Keep `examples/python_problem.py` as runnable callable demo.
  - Make `examples/pure_python_problem.py` runnable with `PythonBuilder`, `.add_parameter`, and `Problem.optimize`.
  - Make `examples/diffsol_problem.py` runnable with the new diffsol builder and HMC sampler.
  - Make `examples/model_evidence.py` runnable (stub evidence is acceptable initially).
- README:
  - Add quickstart showing callable flow, diffsol flow, optimiser and sampler usage, and evidence stub.

### Deliverables and file mapping
- Rust core (`rust/src`):
  - `problem/mod.rs`: add `ProblemKind`, `optimize`, `dimension`, and typed config getters.
  - `problem/diffsol.rs`: new file for `DiffsolModel`, parser, and solver.
  - `samplers/mod.rs` + `samplers/hamiltonian.rs`: new sampler implementation.
- Python bindings (`python/src/lib.rs`):
  - Expose `Problem.optimize`, `Problem.dimension`.
  - Add `Hamiltonian` and `Samples` classes.
  - Add `PythonBuilder` (alias to `Builder`), `.add_parameter`, `.set_optimiser`.
  - Add `builder` submodule with `Diffsol` class.
  - Add `DynamicNestedSampler` (stub) class.
- Examples/Tests:
  - Update all four example scripts to run end-to-end.
  - Add/extend tests in `tests/` for optimiser, sampler, diffsol builder, and evidence stub.

### Acceptance criteria
- `examples/python_problem.py` runs as-is and matches `NelderMead().run(problem, x0)` results.
- `examples/pure_python_problem.py` runs with `PythonBuilder`, `.add_parameter`, and `Problem.optimize`.
- `examples/diffsol_problem.py` runs with `builder.Diffsol` and `Hamiltonian`, yielding `samples.mean_x`.
- `examples/model_evidence.py` runs and prints a numeric `log_z`.
- All public Python classes importable from `chronopt`: `Builder`, `PythonBuilder`, `Problem`, `NelderMead`, `OptimisationResults`, `Hamiltonian`, `Samples`, `builder.Diffsol`, `DynamicNestedSampler`.

### Risks and mitigations
- **DSL parsing scope creep**: start with the minimal grammar to support the example; extend iteratively.
- **HMC stability**: conservative defaults, diagonal mass, validation for NaNs and step size.
- **Parameter ordering**: capture insertion order from Python dict to ensure deterministic mapping to vectors.
- **API drift vs examples**: introduce `PythonBuilder` as alias and small shims (`.add_parameter`, `.set_optimiser`) to satisfy examples quickly while maintaining `Builder`.
