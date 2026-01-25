# Tutorials

Interactive Jupyter notebooks for hands-on learning with Diffid.

## Learning Paths

Follow these progressive learning paths based on your experience level and goals.

### Beginner Track

Perfect for those new to Diffid or optimisation:

<div class="grid cards" markdown>

-   **[Optimisation Basics](notebooks/optimisation_basics.ipynb)**

    ---

    Learn scalar optimisation with the Rosenbrock function. Compare Nelder-Mead, CMA-ES, and Adam optimisers.

    **Topics:** ScalarBuilder, contour plots, optimiser comparison
    **Runtime:** ~5 minutes

-   **[ODE Fitting with DiffSL](notebooks/ode_fitting_diffsol.ipynb)**

    ---

    Fit a logistic growth model to data using DiffSL and Diffsol.

    **Topics:** DiffsolBuilder, DiffSL syntax, parameter fitting
    **Runtime:** ~10 minutes

</div>

### Intermediate Track

Building on the basics with real-world applications:

<div class="grid cards" markdown>

-   **[Parameter Uncertainty](notebooks/parameter_uncertainty.ipynb)**

    ---

    Go from optimisation to MCMC sampling. Quantify parameter uncertainty with confidence intervals.

    **Topics:** Metropolis-Hastings, posterior distributions, diagnostics
    **Runtime:** ~15 minutes

-   **[Model Comparison](notebooks/model_comparison.ipynb)** ⚠️ *Coming Soon*

    ---

    Use Dynamic Nested Sampling to calculate model evidence and Bayes factors.

    **Topics:** Evidence calculation, Bayes factors, model selection
    **Runtime:** ~20 minutes

</div>

## Advanced Track

Complex problems and advanced techniques:

<div class="grid cards" markdown>

-   **[Multi-Backend ODE Solving](notebooks/advanced_predator_prey.ipynb)**

    ---

    Compare Diffsol, JAX/Diffrax, and Julia/DifferentialEquations.jl for predator-prey models.

    **Topics:** VectorBuilder, backend comparison, custom solvers
    **Runtime:** ~20 minutes

</div>

## Quick Reference

| Tutorial                 | Difficulty | Key Concepts              | Prerequisites |
|--------------------------|------------|---------------------------|---------------|
| Optimisation Basics      | ⭐ Beginner | ScalarBuilder, Optimisers | None |
| ODE Fitting              | ⭐ Beginner | DiffsolBuilder, DiffSL    | Optimisation Basics |
| Parameter Uncertainty    | ⭐⭐ Intermediate | MCMC, uncertainty         | Optimisation Basics, ODE Fitting |
| Model Comparison         | ⭐⭐ Intermediate | Nested sampling, evidence | Optimisation Basics, ODE Fitting, Parameter Uncertainty |
| Multi-Backend            | ⭐⭐⭐ Advanced | VectorBuilder, JAX, Julia | Optimisation Basics, ODE Fitting |

## Prerequisites

### All Tutorials
- Python >= 3.11
- Basic Python programming
- NumPy fundamentals
- Jupyter notebook environment

### Additional for Specific Tutorials
- **Parameter Uncertainty & Model Comparison**: Basic Bayesian statistics
- **Multi-Backend**: JAX or Julia knowledge (optional)

## Running the Notebooks

### Installation

Install Diffid with Jupyter and plotting support:

=== "pip"

    ```bash
    pip install diffid jupyter matplotlib
    ```

=== "uv"

    ```bash
    uv pip install diffid jupyter matplotlib
    ```

### Clone and Run

```bash
# Clone the repository
git clone https://github.com/bradyplanden/diffid.git
cd diffid/docs/tutorials/notebooks

# Launch Jupyter
jupyter notebook
```

## Alternative: Python Scripts

Prefer scripts to notebooks? Check out the [examples directory](https://github.com/bradyplanden/diffid/tree/main/examples):

- `python_problem.py` - Basic scalar optimisation
- `logistic_growth.py` - ODE fitting
- `bouncy_ball.py` / `bouncy_ball_sampling.py` - Optimisation and MCMC
- `bicycle_model_evidence.py` - Model comparison
- `predator_prey/` - Multi-backend comparisons

## Contributing

Found an issue or want to improve a tutorial?

1. Fork the [repository](https://github.com/bradyplanden/diffid)
2. Edit notebooks in `docs/tutorials/notebooks/`
3. Test your changes locally
4. Submit a pull request

See the [Contributing Guide](../development/contributing.md) for details.

## Next Steps

After completing the tutorials:

<div class="grid cards" markdown>

-   [:material-book-multiple:{ .lg .middle } __User Guides__](../guides/index.md)

    In-depth guides on choosing and tuning algorithms

-   [:material-code-tags:{ .lg .middle } __API Reference__](../api-reference/index.md)

    Complete API documentation

-   [:material-flask:{ .lg .middle } __Examples Gallery__](../examples/gallery.md)

    More applications and use cases

-   [:material-github:{ .lg .middle } __GitHub Repository__](https://github.com/bradyplanden/diffid)

    Source code and development

</div>
