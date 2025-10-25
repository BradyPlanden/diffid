## Chronopt API (ToDo)

### Likelihoods / Costs
- **Likelihood**
  - General costs with a trait
  - Add a gaussian negative log likelihood implementation

### Phase 1 — Optimisation
  - Add a gradient-based optimization method (AdamW / IRPropMin)
  - Once Hessian information is available, add natural gradient descent implementation

### Phase 2 — Diffsol builder and simulation representation
  - Parallelize the DiffsolProblem with `rayon` (perhaps at OdeBuilder level, need to look into diffsol examples)
  - Implement gradient/hessian acquisition 
  - Implement sparse vs dense solving with faer/Nalgebra

### Phase 3 — Samplers
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

### Phase 4 — Evidence and ergonomics
- **Dynamic nested sampler**
  - `chron.DynamicNestedSampler(problem).run() -> float` returning a placeholder evidence value for now; wire real implementation later or use a simple importance sampling baseline.
- **Initial value inference**
  - Preserve parameter insertion order from Python to construct `x0`.
  - Expose `Problem.dimension()` for sanity checks.
- **Plotting**
  - Add fast contour and Voronoi surface support (via parallelisation where possible)

### Phase 5 — Testing, examples, and docs
- Extend tests:
  - `Problem.optimize()` matches `NelderMead.run(problem, x0)`.
  - `Hamiltonian` returns `mean_x` near MAP on convex objective.
  - ~~Diffsol builder produces a problem that optimizes the logistic example.~~
  - `DynamicNestedSampler.run()` returns a finite value on a toy problem.
- Examples:
  - ~~Keep `examples/python_problem.py` as runnable callable demo.~~
  - ~~Make `examples/pure_python_problem.py` runnable with `PythonBuilder`, `.add_parameter`, and `Problem.optimize`.~~
  - ~~Make `examples/diffsol_problem.py` runnable with the new diffsol builder~~ and HMC sampler.
  - Make `examples/model_evidence.py` runnable (stub evidence is acceptable initially).
- README:
  - ~~Add quickstart showing callable flow, diffsol flow, optimiser and sampler usage, and evidence stub.~~