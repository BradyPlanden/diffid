# Examples Gallery

Visual gallery of Chronopt applications and use cases.

!!! info "Gallery Under Construction"
    This gallery is being populated with examples. Check the [examples directory](https://github.com/bradyplanden/chronopt/tree/main/examples) for current code.

## Available Examples

### Scalar Optimisation

#### Rosenbrock Function
Classic 2D optimisation test problem.

**Files:**

- [python_problem.py](https://github.com/bradyplanden/chronopt/blob/main/examples/python_problem.py)
- [python_contour.py](https://github.com/bradyplanden/chronopt/blob/main/examples/python_contour.py)

**Topics:** ScalarBuilder, contour plots, optimiser comparison

---

### ODE Parameter Fitting

#### Logistic Growth
Single-variable ODE with DiffSL.

**File:** [logistic_growth.py](https://github.com/bradyplanden/chronopt/blob/main/examples/logistic_growth.py)

**Topics:** DiffsolBuilder, DiffSL syntax, data fitting

---

#### Bouncy Ball
Physics-based model with event handling.

**Files:**

- [bouncy_ball.py](https://github.com/bradyplanden/chronopt/blob/main/examples/bouncy_ball.py)
- [bouncy_ball_sampling.py](https://github.com/bradyplanden/chronopt/blob/main/examples/bouncy_ball_sampling.py)

**Topics:** Event detection, parameter uncertainty, MCMC

---

### Model Comparison

#### Bicycle Model
Comparing different bicycle dynamics formulations.

**Files:**

- [bicycle_model_diffsol.py](https://github.com/bradyplanden/chronopt/blob/main/examples/bicycle_model_diffsol.py)
- [bicycle_model_evidence.py](https://github.com/bradyplanden/chronopt/blob/main/examples/bicycle_model_evidence.py)

**Topics:** Model selection, evidence calculation, Bayes factors

---

### Multi-Backend ODE Solving

#### Predator-Prey Models
Lotka-Volterra equations with multiple solver backends.

**Files:**

- [predator_prey_diffsol.py](https://github.com/bradyplanden/chronopt/blob/main/examples/predator_prey/predator_prey_diffsol.py)
- [predator_prey_diffrax.py](https://github.com/bradyplanden/chronopt/blob/main/examples/predator_prey/predator_prey_diffrax.py)
- [predator_prey_diffeqpy.py](https://github.com/bradyplanden/chronopt/blob/main/examples/predator_prey/predator_prey_diffeqpy.py)

**Topics:** VectorBuilder, JAX/Diffrax, Julia/DifferentialEquations.jl, performance comparison

---

## Running Examples

Clone the repository:

```bash
git clone https://github.com/bradyplanden/chronopt.git
cd chronopt
```

Install dependencies:

```bash
pip install chronopt matplotlib
```

Run an example:

```bash
python examples/python_problem.py
```

For ODE examples:

```bash
python examples/logistic_growth.py
```

For multi-backend examples (requires additional dependencies):

```bash
# For Diffrax (JAX)
pip install jax diffrax

# For DifferentialEquations.jl (Julia)
pip install diffeqpy
# Then follow Julia setup instructions

python examples/predator_prey/predator_prey_diffrax.py
```

## Example Categories

### By Problem Type

| Type | Examples | Builder |
|------|----------|---------|
| Scalar | Rosenbrock | ScalarBuilder |
| ODE (DiffSL) | Logistic, Bouncy Ball | DiffsolBuilder |
| ODE (Custom) | Predator-Prey | VectorBuilder |

### By Algorithm

| Algorithm | Examples |
|-----------|----------|
| Nelder-Mead | Most examples (default) |
| CMA-ES | Logistic growth, bicycle model |
| Adam | Coming soon |
| MCMC | Bouncy ball sampling |
| Nested Sampling | Bicycle evidence, model evidence |

### By Difficulty

**Beginner:**

- python_problem.py
- python_contour.py
- logistic_growth.py

**Intermediate:**

- bouncy_ball.py
- bicycle_model_diffsol.py
- predator_prey_diffsol.py

**Advanced:**

- bouncy_ball_sampling.py
- bicycle_model_evidence.py
- predator_prey_diffrax.py
- predator_prey_diffeqpy.py

## Contributing Examples

Have an interesting use case? We'd love to include it!

1. Fork the repository
2. Add your example to `examples/`
3. Include a brief comment header explaining the example
4. Open a pull request

See the [Contributing Guide](../development/contributing.md) for details.

## See Also

- [Tutorials](../tutorials/index.md) - Interactive Jupyter notebooks
- [Getting Started](../getting-started/index.md) - Core concepts
- [API Reference](../api-reference/index.md) - Complete API docs
