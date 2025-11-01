import builtins
import typing

import numpy
import numpy.typing

ObjectiveLike = typing.Callable[[numpy.typing.NDArray[numpy.float64]], builtins.float]
Bounds = tuple[builtins.float, builtins.float]

__all__ = ["contour"]

def contour(
    objective: ObjectiveLike | typing.Any,
    x_bounds: Bounds,
    y_bounds: Bounds,
    *,
    grid_size: builtins.int = ...,
    levels: builtins.int | typing.Iterable[builtins.float] = ...,
    ax: typing.Any | None = ...,
    cmap: builtins.str = ...,
    show: builtins.bool = ...,
    **contour_kwargs: typing.Any,
) -> typing.Any: ...
