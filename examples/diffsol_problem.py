import chronopt as chron

# Example diffsol ODE
ds = """
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
}"""

data = np.linspace(0, 1, 10)
params = {"r": 1.0, "k": 1.0}

# Simple API
config = {"rtol": 1e-6}
builder = (
    chron.builder.Diffsol()
    .add_diffsl(ds)
    .add_data(data)
    .add_config(config)
    .add_params(params)
)
problem = builder.build()
map_result = problem.optimize()

sampler = chron.Hamiltonian()
sampler.set_number_of_chains(6)
sampler.set_parallel(True)

# Run
samples = sampler.run(x0=map_result.x)

print(samples)
print(f"Optimal x: {samples.mean_x}")
