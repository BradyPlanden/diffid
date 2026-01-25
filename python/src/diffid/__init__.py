"""Diffid public Python API."""

from __future__ import annotations

# Error hierarchy
from diffid.errors import (
    AlreadyTerminated,
    BuildError,
    DiffidError,
    EvaluationError,
    ResultCountMismatch,
    TellError,
)

# Plotting module
from diffid import plotting

# Core bindings - optimisers
from diffid._diffid import (
    Adam,
    AdamState,
    CMAES,
    CMAESState,
    NelderMead,
    NelderMeadState,
)

# Core bindings - samplers
from diffid._diffid import (
    DynamicNestedSampler,
    DynamicNestedSamplerState,
    MetropolisHastings,
    MetropolisHastingsState,
    NestedSamples,
    Samples,
)

# Core bindings - builders
from diffid._diffid import (
    DiffsolBuilder,
    ScalarBuilder,
    VectorBuilder,
)

# Core bindings - results and problems
from diffid._diffid import (
    CostMetric,
    Done,
    Evaluate,
    OptimisationResults,
    Problem,
)

# Cost metric factory functions
from diffid._diffid import RMSE as _RMSE
from diffid._diffid import SSE as _SSE
from diffid._diffid import GaussianNLL as _GaussianNLL


def SSE(weight: float = 1.0) -> CostMetric:
    """Sum of Squared Errors cost metric.

    Parameters
    ----------
    weight : float, optional
        Weight multiplier for the cost metric (default: 1.0)

    Returns
    -------
    CostMetric
        Configured SSE cost metric
    """
    return _SSE(weight)


def RMSE(weight: float = 1.0) -> CostMetric:
    """Root Mean Squared Error cost metric.

    Parameters
    ----------
    weight : float, optional
        Weight multiplier for the cost metric (default: 1.0)

    Returns
    -------
    CostMetric
        Configured RMSE cost metric
    """
    return _RMSE(weight)


def GaussianNLL(variance: float = 1.0, weight: float = 1.0) -> CostMetric:
    """Gaussian Negative Log-Likelihood cost metric.

    Parameters
    ----------
    variance : float, optional
        Variance of the Gaussian distribution (default: 1.0)
    weight : float, optional
        Weight multiplier for the cost metric (default: 1.0)

    Returns
    -------
    CostMetric
        Configured Gaussian NLL cost metric

    Notes
    -----
    A single positional argument is interpreted as the variance. An optional
    second positional argument (or keyword) can be used as a weight
    multiplier on the resulting cost.
    """
    return _GaussianNLL(variance, weight)

__all__ = [
    # Modules
    "plotting",
    # Errors
    "DiffidError",
    "EvaluationError",
    "BuildError",
    "TellError",
    "ResultCountMismatch",
    "AlreadyTerminated",
    # Builders (canonical names only)
    "DiffsolBuilder",
    "ScalarBuilder",
    "VectorBuilder",
    # Samplers
    "DynamicNestedSampler",
    "MetropolisHastings",
    "NestedSamples",
    "Samples",
    # Optimisers
    "Adam",
    "CMAES",
    "NelderMead",
    # Results and Problem
    "CostMetric",
    "OptimisationResults",
    "Problem",
    # Cost functions
    "SSE",
    "RMSE",
    "GaussianNLL",
    # Ask-tell types
    "Evaluate",
    "Done",
    # Optimiser states
    "AdamState",
    "CMAESState",
    "NelderMeadState",
    # Sampler states
    "MetropolisHastingsState",
    "DynamicNestedSamplerState",
]
