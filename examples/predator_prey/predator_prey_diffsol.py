import importlib.util
import pathlib

import diffid
import numpy as np

# Example diffsol ODE (logistic growth)
ode = """
in_i { a = 2.0/3.0, b = 4.0/3.0, c = 1.0, d = 1.0 }
x0 { 10.0 } y0 { 5.0 }
u_i {
    y1 = x0,
    y2 = y0,
}
F_i {
    a * y1 - b * y1 * y2,
    c * y1 * y2 - d * y2,
}
"""

# Load shared synthetic dataset (generate if missing)
data_path = pathlib.Path(__file__).with_name("synthetic_data.npz")
if not data_path.exists():
    gen_path = pathlib.Path(__file__).with_name("generate_data_diffrax.py")
    spec = importlib.util.spec_from_file_location("pp_gen", gen_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.main(data_path)
data = np.load(str(data_path))
stacked_data = data["observed_stacked"]

# Simple API
builder = (
    diffid.DiffsolBuilder()
    .with_diffsl(ode)
    .with_data(stacked_data)
    .with_tolerances(rtol=1e-6, atol=1e-8)
    .with_parameter("a", 1.3)
    .with_parameter("b", 0.3)
    .with_parameter("c", 0.05)
    .with_parameter("d", 0.6)
    .with_parallel(True)
    .with_optimiser(
        diffid.NelderMead().with_max_iter(1000)
    )  # Override default optimiser
)
problem = builder.build()

# Optimise
results = problem.optimise()

# Display results
print(results)
print(f"Optimal parameters: {results.x}")
print(f"Optimal cost: {results.value}")
print(f"Optimization success: {results.success}")
print(f"Iterations: {results.iterations}")
print(f"Optimisation time: {results.time}")
print(f"Eval per ms: {results.evaluations / (results.time.total_seconds() * 1000)}")
