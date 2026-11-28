"""
Shape validation helpers and decorators for Tensor operations.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensormist.tensor import Tensor

# ==========================================
# Checks and Guards
# ==========================================


def _validate_shape(shape: tuple[int, ...]) -> None:
    """
    Validates that <shape> is a valid tensor shape.

    An empty tuple is valid and represents a scalar tensor. Boolean values are
    not valid dimensions, even though bool is a subclass of int.

    Raises:
        TypeError: If <shape> is not a tuple of integers.
        ValueError: If <shape> contains invalid dimensions.
    """
    if not isinstance(shape, tuple):
        raise TypeError(
            f"Shape must be a tuple of integers, but received "
            f"{type(shape).__name__}."
        )

    for dim in shape:
        # bool is a subclass of int, so it must be rejected explicitly.
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(
                f"Shape dimensions must be integers, but received "
                f"{type(dim).__name__}."
            )

        if dim < 0:
            raise ValueError(
                f"Shape dimensions must be non-negative, but received {dim}."
            )


def _check_add_shapes(a: Tensor, b: Tensor) -> None:
    """
    Checks whether <a> and <b> have compatible shapes for addition.

    Shapes are compatible when they are equal or when <a> has shape (N, D)
    and <b> has shape (D,).

    Raises:
        TypeError: If <a> or <b> is not a Tensor.
        ValueError: If <a> and <b> have incompatible shapes.
    """
    from tensormist.tensor import Tensor

    if not isinstance(b, Tensor):
        raise TypeError("Both operands must be Tensor instances.")

    if a.shape != b.shape and not (
        a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0]
    ):
        raise ValueError("Operands do not have compatible shapes.")


def _check_equal_shapes(a: Tensor, b: Tensor) -> None:
    """Require a Tensor with exactly the same shape, without broadcasting."""
    from tensormist.tensor import Tensor

    if not isinstance(b, Tensor):
        raise TypeError("Both operands must be Tensor instances.")
    if a.shape != b.shape:
        raise ValueError("Operands must have the same shape.")


def _check_matmul_shapes(a: Tensor, b: Tensor) -> None:
    """
    Checks whether <a> and <b> have compatible shapes for matrix
    multiplication. Both operands must be 2D, and the number of columns in
    <a> must equal the number of rows in <b>.

    Raises:
        TypeError: If <a> or <b> is not a Tensor.
        ValueError: If <a> and <b> have incompatible matrix dimensions.
    """
    from tensormist.tensor import Tensor

    if not isinstance(b, Tensor):
        raise TypeError("Both operands must be Tensor instances.")

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Dimensions of Tensors are incompatible for matrix multiplication.")


# ==========================================
# Wrapper Implementations
# ==========================================


def validates_shape(method: Callable) -> Callable:
    """
    Validate the parameter named <shape> before <method> runs.

    The decorated method must have a parameter named <shape>.
    """
    sig = signature(method)

    @wraps(method)
    def wrapper(*args, **kwargs) -> object:
        bound = sig.bind(*args, **kwargs)
        _validate_shape(bound.arguments["shape"])
        return method(*args, **kwargs)

    return wrapper


def check_add_shapes(method: Callable) -> Callable:
    """Validate addition operand shapes before <method> runs."""

    @wraps(method)
    def wrapper(self: Tensor, x: Tensor) -> Tensor:
        _check_add_shapes(self, x)
        return method(self, x)

    return wrapper


def check_equal_shapes(method: Callable) -> Callable:
    """Validate equal-shape Tensor operands before <method> runs."""

    @wraps(method)
    def wrapper(self: Tensor, x: Tensor) -> Tensor:
        _check_equal_shapes(self, x)
        return method(self, x)

    return wrapper


def check_matmul_shapes(method: Callable) -> Callable:
    """Validate matrix multiplication operand shapes before <method> runs."""

    @wraps(method)
    def wrapper(self: Tensor, x: Tensor) -> Tensor:
        _check_matmul_shapes(self, x)
        return method(self, x)

    return wrapper
