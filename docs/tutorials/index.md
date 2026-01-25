# Tutorials

Interactive Jupyter notebooks for hands-on learning with Diffid.

## Learning Paths

Follow these progressive learning paths based on your experience level and goals.

### 🎯 Beginner Track

Perfect for those new to Diffid or optimisation:

<div class="grid cards" markdown>

-   **[1. Optimisation Basics](notebooks/01_optimization_basics.ipynb)**

    ---

    Learn scalar optimisation with the Rosenbrock function. Compare Nelder-Mead, CMA-ES, and Adam optimisers.

    **Topics:** ScalarBuilder, contour plots, optimiser comparison
    **Runtime:** ~5 minutes

-   **[2. ODE Fitting with DiffSL](notebooks/02_ode_fitting_diffsol.ipynb)**

    ---

    Fit a logistic growth model to data using DiffSL and Diffsol.

    **Topics:** DiffsolBuilder, DiffSL syntax, parameter fitting
    **Runtime:** ~10 minutes

</div>

### 🚀 Intermediate Track

Building on the basics with real-world applications:

<div class="grid cards" markdown>

-   **[3. Parameter Uncertainty](notebooks/03_parameter_uncertainty.ipynb)**

    ---

    Go from optimisation to MCMC sampling. Quantify parameter uncertainty with confidence intervals.

    **Topics:** Metropolis-Hastings, posterior distributions, diagnostics
    **Runtime:** ~15 minutes

-   **[4. Model Comparison](notebooks/04_model_comparison.ipynb)** ⚠️ *Coming Soon*

    ---

    Use Dynamic Nested Sampling to calculate model evidence and Bayes factors.

    **Topics:** Evidence calculation, Bayes factors, model selection
    **Runtime:** ~20 minutes

</div>

### 🔬 Advanced Track

Complex problems and advanced techniques:

<div class="grid cards" markdown>

-   **[5. Multi-Backend ODE Solving](notebooks/05_advanced_predator_prey.ipynb)**

    ---

    Compare Diffsol, JAX/Diffrax, and Julia/DifferentialEquations.jl for predator-prey models.

    **Topics:** VectorBuilder, backend comparison, custom solvers
    **Runtime:** ~20 minutes

</div>

## Quick Reference

| Tutorial | Difficulty | Key Concepts | Prerequisites |
|----------|------------|--------------|---------------|
| 1. Optimization Basics | ⭐ Beginner | ScalarBuilder, optimizers | None |
| 2. ODE Fitting | ⭐ Beginner | DiffsolBuilder, DiffSL | Tutorial 1 |
| 3. Parameter Uncertainty | ⭐⭐ Intermediate | MCMC, uncertainty | Tutorials 1-2 |
| 4. Model Comparison | ⭐⭐ Intermediate | Nested sampling, evidence | Tutorials 1-3 |
| 5. Multi-Backend | ⭐⭐⭐ Advanced | VectorBuilder, JAX, Julia | Tutorials 1-2 |

## Prerequisites

### All Tutorials
- Python >= 3.11
- Basic Python programming
- NumPy fundamentals
- Jupyter notebook environment

### Additional for Specific Tutorials
- **Tutorials 3-4**: Basic Bayesian statistics
- **Tutorial 5**: JAX or Julia knowledge (optional)

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

### Google Colab

You can also run these notebooks in Google Colab (coming soon with hosted versions).

## Learning Outcomes

By completing all tutorials, you will:

- Understand Diffid's builder pattern and API
- Optimize both scalar functions and ODE parameters
- Compare and tune different optimisation algorithms
- Quantify parameter uncertainty with MCMC
- Compare models using Bayesian evidence
- Integrate custom ODE solvers (JAX, Julia)
- Make informed decisions about algorithm selection

## Notebook Structure

Each tutorial follows a consistent structure:

1. **Learning Objectives** - What you'll learn
2. **Prerequisites** - Required background
3. **Introduction** - Problem context and motivation
4. **Step-by-Step Code** - Fully explained examples
5. **Visualizations** - Plots and diagnostics
6. **Key Takeaways** - Summary of main points
7. **Exercises** - Practice problems
8. **Next Steps** - Links to related content

## Alternative: Python Scripts

Prefer scripts to notebooks? Check out the [examples directory](https://github.com/bradyplanden/diffid/tree/main/examples):

- `python_problem.py` - Basic scalar optimisation
- `logistic_growth.py` - ODE fitting
- `bouncy_ball.py` / `bouncy_ball_sampling.py` - Optimisation and MCMC
- `bicycle_model_evidence.py` - Model comparison
- `predator_prey/` - Multi-backend comparisons

## Utilities

The notebooks use shared utilities in `utils.py`:

- `plot_contour_2d()` - 2D function contours
- `plot_ode_fit()` - ODE fits and data
- `plot_convergence()` - Optimisation history
- `plot_parameter_traces()` - MCMC traces
- `plot_parameter_distributions()` - Posterior histograms
- `compare_models()` - Multi-model plots

Feel free to reuse these in your own projects!

## Troubleshooting

### Import Errors

```python
ModuleNotFoundError: No module named 'diffid'
```

**Solution**: Install Diffid: `pip install diffid`

### Notebook Kernel Issues

If the notebook doesn't recognize installed packages:

1. Install packages in the correct environment
2. Restart the Jupyter kernel: *Kernel → Restart*
3. Check kernel selection: *Kernel → Change Kernel*

### Missing Dependencies

Some notebooks require optional dependencies:

```bash
# For plotting
pip install matplotlib

# For Tutorial 5 (optional backends)
pip install jax diffrax        # JAX/Diffrax
pip install diffeqpy           # Julia
```

### Performance Issues

If MCMC sampling is slow:

- Reduce number of chains or iterations
- Enable parallel execution: `.with_parallel(True)`
- Use fewer data points for testing

## Getting Help

- **Documentation**: Browse the [complete docs](../index.md)
- **Examples**: See the [examples gallery](../examples/gallery.md)
- **API Reference**: Check the [API docs](../api-reference/index.md)
- **Issues**: Report problems on [GitHub](https://github.com/bradyplanden/diffid/issues)

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
