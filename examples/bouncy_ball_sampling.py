import diffid
import numpy as np


def ball_states(t: np.ndarray, g: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    height = h - 0.5 * g * np.square(t)
    height = np.maximum(height, 0.0)
    velocity = -g * t
    return height, velocity


# DiffSL program for a falling (bouncy) ball terminated when the height reaches zero.
dsl = """
in_i { g = 2.5, h = 1 }
u_i {x = h, v = 0}
F_i {v, -g}
stop {x}
"""

# Data setup
g_true = 9.81
h_true = 10.0
t_stop = np.sqrt(2.0 * h_true / g_true)
t_final = 0.7 * t_stop
t_span = np.linspace(0.0, t_final, 61)
height, velocity = ball_states(t_span, g_true, h_true)
noise = np.random.normal(0, 0.01, len(t_span))
data = np.column_stack((t_span, height + noise, velocity + noise))

# Configure the problem
initial_values = [4.0, 4.0]
builder = (
    diffid.DiffsolBuilder()
    .with_diffsl(dsl)
    .with_data(data)
    .with_parameter("g", initial_values[0])
    .with_parameter("h", initial_values[1])
    .with_parallel(True)
    .with_cost(diffid.GaussianNLL(variance=0.01))
)

problem = builder.build()

# Setup sampler
sampler = (
    diffid.MetropolisHastings()
    .with_num_chains(100)
    .with_iterations(1000)
    .with_step_size(0.25)
    .with_seed(1234)
)

samples = sampler.run(problem, initial=initial_values)
print(samples)
