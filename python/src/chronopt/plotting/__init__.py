"""Plotting utilities for Chronopt.

This module provides convenience helpers for visualising optimisation and sampling
results. The implementation only depends on ``numpy`` at import time and lazily
imports ``matplotlib`` when required so that plotting remains an optional
dependency of the project.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from chronopt import Problem

__all__ = [
    "contour",
    "contour_2d",
    "ode_fit",
    "convergence",
    "parameter_traces",
    "parameter_distributions",
    "compare_models",
]

ObjectiveLike = Union[Callable[[Sequence[float]], float], Callable[[np.ndarray], float]]
Bounds = tuple[float, float]


def _setup_plotting() -> None:
    """Configure matplotlib for nice-looking plots."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    plt.rcParams.setdefault("figure.figsize", (10, 6))
    plt.rcParams.setdefault("font.size", 11)
    plt.rcParams.setdefault("axes.labelsize", 12)
    plt.rcParams.setdefault("axes.titlesize", 14)
    plt.rcParams.setdefault("legend.fontsize", 10)
    plt.rcParams.setdefault("xtick.labelsize", 10)
    plt.rcParams.setdefault("ytick.labelsize", 10)


def _evaluate(objective: ObjectiveLike | Problem, point: np.ndarray) -> float:
    if hasattr(objective, "evaluate"):
        return float(objective.evaluate(point.tolist()))
    return float(objective(point))


def contour(
    objective: ObjectiveLike | Problem,
    x_bounds: Bounds,
    y_bounds: Bounds,
    *,
    grid_size: int = 100,
    levels: int | Iterable[float] = 10,
    ax: Any | None = None,
    cmap: str = "viridis",
    show: bool = True,
    **contour_kwargs: Any,
) -> Any:
    """Render a contour plot of a two-dimensional objective.

    Parameters
    ----------
    objective:
        A callable mapping a two-dimensional input to a scalar value, or a
        :class:`chronopt.Problem` instance whose ``evaluate`` method will be
        invoked.
    x_bounds, y_bounds:
        Inclusive ranges ``(min, max)`` spanning the region to sample along each
        axis.
    grid_size:
        Number of sample points per axis used to generate the evaluation grid.
    levels:
        Either the number of contour levels, or an explicit iterable of level
        values passed through to :func:`matplotlib.pyplot.contour`.
    ax:
        Optional existing matplotlib axes to draw on. If omitted, a new figure
        and axes are created.
    cmap:
        Name of the matplotlib colormap to use for the contour lines.
    show:
        Whether to call :func:`matplotlib.pyplot.show` after drawing the plot.
    **contour_kwargs:
        Additional keyword arguments forwarded to ``Axes.contour``.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The contour set created by matplotlib.

    Raises
    ------
    ValueError
        If ``grid_size`` is not greater than one or if the provided bounds are
        invalid.
    ModuleNotFoundError
        If ``matplotlib`` is not installed in the current environment.
    """

    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Bounds must satisfy min < max along both axes")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; install it via 'pip install chronopt[plotting]'"
        ) from exc

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)

    stacked = np.stack([grid_x, grid_y], axis=-1)
    flattened = stacked.reshape(-1, 2)
    evaluated = np.array([_evaluate(objective, point) for point in flattened]).reshape(
        grid_size, grid_size
    )

    if ax is None:
        _, ax = plt.subplots()

    contour_set = ax.contour(grid_x, grid_y, evaluated, levels=levels, cmap=cmap, **contour_kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Objective contour")
    ax.clabel(contour_set, inline=True, fontsize=8)

    if show:
        plt.show()

    return contour_set


def contour_2d(
    func: Callable,
    xlim: tuple[float, float] = (-2, 2),
    ylim: tuple[float, float] = (-1, 3),
    *,
    levels: np.ndarray | None = None,
    optimum: tuple[float, float] | None = None,
    found: tuple[float, float] | None = None,
    title: str = "Optimisation Landscape",
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot 2D function contours with optional optimum markers.

    This is a convenience wrapper for notebook usage that provides a simplified
    interface for plotting objective functions with marked optima.

    Parameters
    ----------
    func:
        Function that takes [x, y] and returns scalar value.
    xlim:
        x-axis limits (min, max).
    ylim:
        y-axis limits (min, max).
    levels:
        Contour levels to plot (optional).
    optimum:
        True optimum location (x, y) to mark (optional).
    found:
        Found optimum location (x, y) to mark (optional).
    title:
        Plot title.
    show:
        Whether to call :func:`matplotlib.pyplot.show` after drawing the plot.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.

    Raises
    ------
    ModuleNotFoundError
        If ``matplotlib`` is not installed in the current environment.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; install it via 'pip install chronopt[plotting]'"
        ) from exc

    _setup_plotting()

    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result = func([X[i, j], Y[i, j]])
            # Handle both scalar and array-like returns
            Z[i, j] = result[0] if hasattr(result, "__getitem__") else result

    fig, ax = plt.subplots(figsize=(10, 8))

    if levels is None:
        levels = np.logspace(-1, 3.5, 20)

    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8)

    if optimum:
        ax.plot(
            optimum[0],
            optimum[1],
            "r*",
            markersize=20,
            label="Global minimum",
            zorder=5,
        )

    if found:
        ax.plot(
            found[0], found[1], "go", markersize=10, label="Found optimum", zorder=5
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


def ode_fit(
    t_data: np.ndarray,
    y_data: np.ndarray,
    t_pred: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    *,
    title: str = "ODE Fit",
    xlabel: str = "Time",
    ylabel: str = "State Variable",
    ax: Any | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot ODE data and fitted model.

    Parameters
    ----------
    t_data:
        Time points for observed data.
    y_data:
        Observed data values.
    t_pred:
        Time points for predictions (optional).
    y_pred:
        Predicted values (optional).
    title:
        Plot title.
    xlabel:
        x-axis label.
    ylabel:
        y-axis label.
    ax:
        Optional existing matplotlib axes to draw on. If omitted, a new figure
        and axes are created.
    show:
        Whether to call :func:`matplotlib.pyplot.show` after drawing the plot.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.

    Raises
    ------
    ModuleNotFoundError
        If ``matplotlib`` is not installed in the current environment.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; install it via 'pip install chronopt[plotting]'"
        ) from exc

    _setup_plotting()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(t_data, y_data, "o", label="Observed data", alpha=0.6, markersize=8)

    if t_pred is not None and y_pred is not None:
        ax.plot(t_pred, y_pred, "-", label="Fitted model", linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig, ax


def convergence(
    values: np.ndarray,
    *,
    title: str = "Optimisation Convergence",
    log_scale: bool = True,
    ax: Any | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot optimisation convergence history.

    Parameters
    ----------
    values:
        Objective function values over iterations.
    title:
        Plot title.
    log_scale:
        Use log scale for y-axis.
    ax:
        Optional existing matplotlib axes to draw on. If omitted, a new figure
        and axes are created.
    show:
        Whether to call :func:`matplotlib.pyplot.show` after drawing the plot.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.

    Raises
    ------
    ModuleNotFoundError
        If ``matplotlib`` is not installed in the current environment.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; install it via 'pip install chronopt[plotting]'"
        ) from exc

    _setup_plotting()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(values, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    if show:
        plt.show()

    return fig, ax


def parameter_traces(
    samples: np.ndarray,
    param_names: Sequence[str],
    true_values: Sequence[float] | None = None,
    *,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot MCMC parameter traces.

    Parameters
    ----------
    samples:
        MCMC samples with shape (n_samples, n_params).
    param_names:
        Parameter names.
    true_values:
        True parameter values to mark (optional).
    show:
        Whether to call :func:`matplotlib.pyplot.show` after drawing the plot.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The matplotlib figure and list of axes objects.

    Raises
    ------
    ModuleNotFoundError
        If ``matplotlib`` is not installed in the current environment.
    ValueError
        If the number of parameter names doesn't match sample dimensions.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; install it via 'pip install chronopt[plotting]'"
        ) from exc

    _setup_plotting()

    n_params = samples.shape[1]
    if len(param_names) != n_params:
        raise ValueError(
            f"Number of parameter names ({len(param_names)}) must match "
            f"sample dimensions ({n_params})"
        )

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params))

    if n_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(samples[:, i], alpha=0.7, linewidth=0.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

        if true_values and i < len(true_values):
            ax.axhline(
                true_values[i],
                color="r",
                linestyle="--",
                label=f"True value: {true_values[i]:.3f}",
            )
            ax.legend()

        if i == n_params - 1:
            ax.set_xlabel("Sample")

    fig.suptitle("MCMC Parameter Traces")
    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def parameter_distributions(
    samples: np.ndarray,
    param_names: Sequence[str],
    true_values: Sequence[float] | None = None,
    *,
    bins: int = 30,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot posterior parameter distributions.

    Parameters
    ----------
    samples:
        MCMC samples with shape (n_samples, n_params).
    param_names:
        Parameter names.
    true_values:
        True parameter values to mark (optional).
    bins:
        Number of histogram bins.
    show:
        Whether to call :func:`matplotlib.pyplot.show` after drawing the plot.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The matplotlib figure and list of axes objects.

    Raises
    ------
    ModuleNotFoundError
        If ``matplotlib`` is not installed in the current environment.
    ValueError
        If the number of parameter names doesn't match sample dimensions.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; install it via 'pip install chronopt[plotting]'"
        ) from exc

    _setup_plotting()

    n_params = samples.shape[1]
    if len(param_names) != n_params:
        raise ValueError(
            f"Number of parameter names ({len(param_names)}) must match "
            f"sample dimensions ({n_params})"
        )

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples[:, i], bins=bins, density=True, alpha=0.7, edgecolor="black")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        if true_values and i < len(true_values):
            ax.axvline(
                true_values[i],
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"True: {true_values[i]:.3f}",
            )

        # Add mean and std
        mean = np.mean(samples[:, i])
        std = np.std(samples[:, i])
        ax.axvline(
            mean,
            color="blue",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {mean:.3f}",
        )
        ax.set_title(f"{name}\n(σ = {std:.3f})")
        ax.legend()

    fig.suptitle("Parameter Posterior Distributions")
    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def compare_models(
    t_data: np.ndarray,
    y_data: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str = "Model Comparison",
    ax: Any | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot multiple model predictions against data.

    Parameters
    ----------
    t_data:
        Time points for observed data.
    y_data:
        Observed data.
    predictions:
        Dictionary mapping model names to (t_pred, y_pred) tuples.
    title:
        Plot title.
    ax:
        Optional existing matplotlib axes to draw on. If omitted, a new figure
        and axes are created.
    show:
        Whether to call :func:`matplotlib.pyplot.show` after drawing the plot.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.

    Raises
    ------
    ModuleNotFoundError
        If ``matplotlib`` is not installed in the current environment.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting; install it via 'pip install chronopt[plotting]'"
        ) from exc

    _setup_plotting()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(
        t_data, y_data, "ko", label="Observed data", alpha=0.6, markersize=8, zorder=5
    )

    for name, (t_pred, y_pred) in predictions.items():
        ax.plot(t_pred, y_pred, "-", label=name, linewidth=2, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("State Variable")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig, ax
