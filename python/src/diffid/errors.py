"""Custom exception hierarchy for Diffid.

This module defines the exception hierarchy used throughout the diffid library,
providing clear and actionable error messages for different failure modes.
"""

from __future__ import annotations


class DiffidError(Exception):
    """Base exception class for all Diffid errors.

    All custom exceptions in the diffid library inherit from this class,
    making it easy to catch any diffid-specific error.

    Examples
    --------
    >>> try:
    ...     optimiser.run(problem, initial=[1.0, 2.0])
    ... except DiffidError as e:
    ...     print(f"Diffid error occurred: {e}")
    """

    pass


class EvaluationError(DiffidError):
    """Raised when objective function evaluation fails.

    This exception is raised when the objective function (or callback) throws
    an exception during evaluation, or when evaluation produces invalid values
    (NaN, infinite, etc.).

    Parameters
    ----------
    message : str
        Description of the evaluation failure
    point : list[float] | None, optional
        The parameter values at which evaluation failed
    original_error : Exception | None, optional
        The original exception that caused the evaluation failure

    Examples
    --------
    >>> def problematic_objective(x):
    ...     if x[0] < 0:
    ...         raise ValueError("x[0] must be positive")
    ...     return sum(xi**2 for xi in x)
    >>>
    >>> try:
    ...     optimiser.run(problem, initial=[-1.0, 2.0])
    ... except EvaluationError as e:
    ...     print(f"Evaluation failed: {e}")
    ...     print(f"At point: {e.point}")
    """

    def __init__(
        self,
        message: str,
        point: list[float] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.point = point
        self.original_error = original_error


class BuildError(DiffidError):
    """Raised when problem or optimiser construction fails.

    This exception is raised during the build phase when invalid parameters
    are provided, or when the configuration is inconsistent.

    Examples
    --------
    >>> try:
    ...     builder = ScalarBuilder()
    ...     builder.with_iterations(-10)  # Invalid: negative iterations
    ...     problem = builder.build(lambda x: x**2)
    ... except BuildError as e:
    ...     print(f"Build failed: {e}")
    """

    pass


class TellError(DiffidError):
    """Base exception for errors during the 'tell' phase of ask-tell interface.

    This exception is raised when providing results back to an optimiser or
    sampler state after being asked for evaluations.

    See Also
    --------
    ResultCountMismatch : When number of results doesn't match requested points
    AlreadyTerminated : When attempting to continue a terminated optimization
    """

    pass


class ResultCountMismatch(TellError):
    """Raised when number of results doesn't match number of requested points.

    During the ask-tell interface, after calling `ask()` which returns N points
    to evaluate, `tell()` must be called with exactly N results. This exception
    is raised if the counts don't match.

    Parameters
    ----------
    expected : int
        Number of results expected (number of points from ask())
    received : int
        Number of results actually provided to tell()

    Examples
    --------
    >>> state = optimiser.init(problem, initial=[1.0, 2.0])
    >>> query = state.ask()
    >>> if isinstance(query, Evaluate):
    ...     # Wrong: only providing 1 result when multiple points were requested
    ...     try:
    ...         state.tell([problem(query.points[0])])
    ...     except ResultCountMismatch as e:
    ...         print(f"Expected {e.expected} results, got {e.received}")
    """

    def __init__(self, expected: int, received: int):
        message = f"Expected {expected} evaluation results, but received {received}"
        super().__init__(message)
        self.expected = expected
        self.received = received


class AlreadyTerminated(TellError):
    """Raised when attempting to continue an already terminated optimization.

    Once an optimization or sampling run has completed (ask() returns Done),
    no further tell() calls should be made. This exception is raised if you
    attempt to continue a terminated state.

    Examples
    --------
    >>> state = optimiser.init(problem, initial=[1.0, 2.0])
    >>> while True:
    ...     query = state.ask()
    ...     if isinstance(query, Done):
    ...         result = query.result
    ...         break
    ...     # ... evaluate and tell
    >>>
    >>> # This will raise AlreadyTerminated:
    >>> try:
    ...     state.tell([0.0])
    ... except AlreadyTerminated as e:
    ...     print(f"Cannot continue: {e}")
    """

    def __init__(self):
        super().__init__("Cannot provide results to an already terminated optimization")


__all__ = [
    "DiffidError",
    "EvaluationError",
    "BuildError",
    "TellError",
    "ResultCountMismatch",
    "AlreadyTerminated",
]
