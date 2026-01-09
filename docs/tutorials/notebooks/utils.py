"""Utility functions for Chronopt tutorial notebooks."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


def setup_plotting():
    """Configure matplotlib for nice-looking plots."""
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def plot_contour_2d(
    func: Callable,
    xlim: tuple = (-2, 2),
    ylim: tuple = (-1, 3),
    levels: np.ndarray | None = None,
    optimum: tuple | None = None,
    found: tuple | None = None,
    title: str = "Optimisation Landscape",
):
    """
    Plot 2D function contours with optional optimum markers.

    Parameters
    ----------
    func : callable
        Function that takes [x, y] and returns scalar value
    xlim : tuple
        x-axis limits (min, max)
    ylim : tuple
        y-axis limits (min, max)
    levels : array-like, optional
        Contour levels to plot
    optimum : tuple, optional
        True optimum location (x, y)
    found : tuple, optional
        Found optimum location (x, y)
    title : str
        Plot title
    """
    setup_plotting()

    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])[0]

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

    return fig, ax


def plot_ode_fit(
    t_data: np.ndarray,
    y_data: np.ndarray,
    t_pred: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    title: str = "ODE Fit",
    xlabel: str = "Time",
    ylabel: str = "State Variable",
):
    """
    Plot ODE data and fitted model.

    Parameters
    ----------
    t_data : array
        Time points for observed data
    y_data : array
        Observed data values
    t_pred : array, optional
        Time points for predictions
    y_pred : array, optional
        Predicted values
    title : str
        Plot title
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    """
    setup_plotting()

    fig, ax = plt.subplots()

    ax.plot(t_data, y_data, "o", label="Observed data", alpha=0.6, markersize=8)

    if t_pred is not None and y_pred is not None:
        ax.plot(t_pred, y_pred, "-", label="Fitted model", linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_convergence(
    values: np.ndarray, title: str = "Optimisation Convergence", log_scale: bool = True
):
    """
    Plot optimisation convergence history.

    Parameters
    ----------
    values : array
        Objective function values over iterations
    title : str
        Plot title
    log_scale : bool
        Use log scale for y-axis
    """
    setup_plotting()

    fig, ax = plt.subplots()

    ax.plot(values, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    return fig, ax


def plot_parameter_traces(
    samples: np.ndarray, param_names: list, true_values: list | None = None
):
    """
    Plot MCMC parameter traces.

    Parameters
    ----------
    samples : array
        MCMC samples (n_samples, n_params)
    param_names : list
        Parameter names
    true_values : list, optional
        True parameter values to mark
    """
    setup_plotting()

    n_params = samples.shape[1]
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

    return fig, axes


def plot_parameter_distributions(
    samples: np.ndarray, param_names: list, true_values: list | None = None
):
    """
    Plot posterior parameter distributions.

    Parameters
    ----------
    samples : array
        MCMC samples (n_samples, n_params)
    param_names : list
        Parameter names
    true_values : list, optional
        True parameter values to mark
    """
    setup_plotting()

    n_params = samples.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples[:, i], bins=30, density=True, alpha=0.7, edgecolor="black")
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
            ax.legend()

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

    fig.suptitle("Parameter Posterior Distributions")
    plt.tight_layout()

    return fig, axes


def compare_models(
    t_data: np.ndarray,
    y_data: np.ndarray,
    predictions: dict,
    title: str = "Model Comparison",
):
    """
    Plot multiple model predictions against data.

    Parameters
    ----------
    t_data : array
        Time points for observed data
    y_data : array
        Observed data
    predictions : dict
        Dictionary of {model_name: (t_pred, y_pred)}
    title : str
        Plot title
    """
    setup_plotting()

    fig, ax = plt.subplots()

    ax.plot(
        t_data, y_data, "ko", label="Observed data", alpha=0.6, markersize=8, zorder=5
    )

    for i, (name, (t_pred, y_pred)) in enumerate(predictions.items()):
        ax.plot(t_pred, y_pred, "-", label=name, linewidth=2, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("State Variable")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
