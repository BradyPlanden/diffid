"""Plotting utilities for Chronopt.

This module currently provides a convenience helper to visualise two-dimensional
objective functions via contour plots. The implementation only depends on
``numpy`` at import time and lazily imports ``matplotlib`` when required so that
plotting remains an optional dependency of the project.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from chronopt import Problem

__all__ = ["contour"]

ObjectiveLike = Union[Callable[[Sequence[float]], float], Callable[[np.ndarray], float]]
Bounds = tuple[float, float]


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
