# import chronopt as chron
# import numpy as np
#
#
# # Build an optimisation problem
# def rosenbrock(x):
#     return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
#
#
# def skip_optimisation_test():
#     data = np.linspace(0, 10, 10)
#     theta = bernoulli(0.2, 0.8)
#     builder = (
#         chron.builders.Problem()
#         .add_callable(rosenbrock)
#         .add_parameter(theta)
#         .add_data(data)
#     )
#     builder.build()
#
#     # Create the optimisation
#     optim = chron.optim.CMAES()
#     optim.set_sigma0(1.0)  # Initial covariance
#     optim.set_time_allowance(60.0)  # Optimiser time limit in seconds
#     optim.set_convergence_threshold(
#         1e-5
#     )  # threshold to start unchanged iteration count
#     optim.set_patience(
#         10
#     )  # number of iterations at threshold until finishing optimisation
#
#     results = optim.run()
#
#     # Validation metrics
#     # Some may require additional arguments, or further computation
#     results.hessian()  # Finite-diff, covariance from CMAES, autodiff
#     results.evidence()
#     results.sensitivities()
#     results.uncertainties()
