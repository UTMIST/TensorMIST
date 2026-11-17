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
    # TODO
    raise NotImplementedError


def _check_add_shapes(a: Tensor, b: Tensor) -> None:
    """
    Checks whether <a> and <b> have compatible shapes for addition.

    Shapes are compatible when they are equal or when <a> has shape (N, D)
    and <b> has shape (D,).

    Raises:
        TypeError: If <a> or <b> is not a Tensor.
        ValueError: If <a> and <b> have incompatible shapes.
    """
    if not isinstance(b, type(a)):
        raise TypeError("Both operands must be Tensor instances.")

    # TODO
    raise NotImplementedError


def _check_matmul_shapes(a: Tensor, b: Tensor) -> None:
    """
    Checks whether <a> and <b> have compatible shapes for matrix
    multiplication. Both operands must be 2D, and the number of columns in
    <a> must equal the number of rows in <b>.

    Raises:
        TypeError: If <a> or <b> is not a Tensor.
        ValueError: If <a> and <b> have incompatible matrix dimensions.
    """
    if not isinstance(b, type(a)):
        raise TypeError("Both operands must be Tensor instances.")

    # TODO
    raise NotImplementedError


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


def check_matmul_shapes(method: Callable) -> Callable:
    """Validate matrix multiplication operand shapes before <method> runs."""

    @wraps(method)
    def wrapper(self: Tensor, x: Tensor) -> Tensor:
        _check_matmul_shapes(self, x)
        return method(self, x)

    return wrapper
