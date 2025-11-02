import chronopt as chron
import numpy as np


def ball_states(t: np.ndarray, g: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    height = h - 0.5 * g * np.square(t)
    height = np.maximum(height, 0.0)
    velocity = -g * t
    return height, velocity


def test_diffsol_sampling_tracks_bouncy_ball_parameters():
    # DiffSL program for a falling (bouncy) ball terminated when the height reaches zero.
    dsl = """
in = [g, h]
g { 1 } h { 1 }
u_i {x = h, v = 0}
F_i {v, -g}
stop {x}
"""

    g_true = 9.81
    h_true = 10.0
    t_stop = np.sqrt(2.0 * h_true / g_true)
    t_final = 0.7 * t_stop
    t_span = np.linspace(0.0, t_final, 61)
    height, velocity = ball_states(t_span, g_true, h_true)
    data = np.column_stack((t_span, height, velocity))

    builder = (
        chron.DiffsolBuilder()
        .add_diffsl(dsl)
        .add_data(data)
        .add_params({"g": g_true, "h": h_true})
        .with_rtol(1e-6)
        .with_atol(1e-6)
        .add_cost(chron.costs.SSE())
    )

    problem = builder.build()

    initial_guess = [8.0, 8.0]
    initial_cost = problem.evaluate(initial_guess)

    sampler = (
        chron.samplers.MetropolisHastings()
        .with_num_chains(2)
        .with_num_steps(250)
        .with_step_size(0.25)
        .with_seed(1234)
    )

    samples = sampler.run(problem, initial_guess)

    assert samples.draws == 2 * 250
    assert len(samples.chains) == 2
    assert all(len(chain) == 251 for chain in samples.chains)

    burn_in = 50
    post_burn_in = [sample for chain in samples.chains for sample in chain[burn_in:]]
    assert post_burn_in, "Sampler should produce post burn-in samples"

    post_costs = [problem.evaluate(sample) for sample in post_burn_in]
    assert min(post_costs) < initial_cost, "Samples should improve on initial cost"

    mean_estimate = np.mean(post_burn_in, axis=0)
    assert np.all(np.isfinite(mean_estimate))
    assert abs(mean_estimate[0] - g_true) < 0.1
    assert abs(mean_estimate[1] - h_true) < 0.1
