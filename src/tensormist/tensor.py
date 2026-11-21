"""
An implementation of a tensor object and its supported operations.
"""

from __future__ import annotations

import array
from collections.abc import Iterable
from math import prod

from tensormist.validation import check_add_shapes, check_matmul_shapes, validates_shape


class Tensor:
    """
    Represents a multidimensional array of floating-point values.

    A Tensor stores its values in a flat sequence together with a shape that
    describes the dimensions of the array. It supports tensor metadata access,
    indexed access, and arithmetic operations.
    """

    @validates_shape
    def __init__(self, data: Iterable[float] | array.array, shape: tuple[int, ...]):
        """
        Initializes the tensor with shape <shape> and <data> preloaded.

        Inputs:
            data: An iterable of floating-point values, or an
                `array.array('f')` with `itemsize == 4`, used to initialize
                the tensor.
            shape: A tuple of non-negative integer dimensions describing the
                tensor shape.

        Outputs:
            None. Initializes this Tensor instance in place.

        Raises:
            TypeError: If <data> or <shape> has an unsupported type.
            ValueError: If <shape> contains invalid dimensions or the number
                of values in <data> does not match the shape.
            RuntimeError: If <data> is an array with a type code other than
                `'f'` or an item size other than 4 bytes.

        Notes:
            Shape validation is performed by the shape decorator. The method
            body is responsible for validating <data>, its length, and its
            float32 storage requirements. If array.array is used in the
            initialisation, its elements will be copied.
        """

        # Data: Require a numeric iterable and copy its values into private
        # contiguous, row-major float32 storage.
        if isinstance(data, (str, bytes, bytearray)) or not isinstance(data, Iterable):
            raise TypeError("Data must be an iterable of numbers.")

        if isinstance(data, array.array):
            if data.typecode != 'f' or data.itemsize != 4:
                raise RuntimeError("Array must use 4-byte float32 storage.")

        try:
            self._data = array.array('f', data)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("Data must contain valid numeric values.") from exc

        # Validation: Check the float32 element size and ensure the number
        # of stored elements matches the product of the shape dimensions.
        if self._data.itemsize != 4:
            raise RuntimeError("Float32 must occupy 4 bytes.")

        expected_numel = prod(shape)

        if len(self._data) != expected_numel:
            raise ValueError(
                f"Expected {expected_numel} elements, "
                f"but received {len(self._data)}."
            )

        # Shape: Store the tensor dimensions as metadata.
        # General dimension validation is handled by @validates_shape.
        self._shape = shape

    # ==========================================
    # Special Initializations
    # ==========================================

    @classmethod
    @validates_shape
    def ones(cls, shape: tuple[int, ...]) -> Tensor:
        """
        Creates a tensor with the specified shape, initialized with ones.

        Inputs:
            shape: A tuple of non-negative integer dimensions describing the
                tensor shape.

        Outputs:
            A Tensor with shape <shape> and every element initialized to 1.0.

        Raises:
            TypeError: If <shape> has an unsupported type.
            ValueError: If <shape> contains invalid dimensions.
        """
        return cls._from_array(array.array('f', [1.0]) * prod(shape), shape)

    @classmethod
    @validates_shape
    def zeros(cls, shape: tuple[int, ...]) -> Tensor:
        """
        Creates a tensor with the specified shape, initialized with zeros.

        Inputs:
            shape: A tuple of non-negative integer dimensions describing the
                tensor shape.

        Outputs:
            A Tensor with shape <shape> and every element initialized to 0.0.

        Raises:
            TypeError: If <shape> has an unsupported type.
            ValueError: If <shape> contains invalid dimensions.
        """
        return cls._from_array(array.array('f', [0.0]) * prod(shape), shape)

    @classmethod
    @validates_shape
    def _from_array(cls, data: array.array, shape: tuple[int, ...]) -> Tensor:
        """
        Creates a tensor that takes ownership of <data> without copying it.

        Inputs:
            data: An `array.array('f')` with `itemsize == 4` holding exactly
                prod(<shape>) values in row-major order.
            shape: A tuple of non-negative integer dimensions describing the
                tensor shape.

        Outputs:
            A Tensor with shape <shape> whose storage is <data> itself.

        Raises:
            TypeError: If <data> is not an array.array.
            ValueError: If <shape> contains invalid dimensions or the length
                of <data> does not match the shape.
            RuntimeError: If <data> does not use 4-byte float32 storage.

        Notes:
            Internal fast path for ones(), zeros(), and operation outputs.
            The caller must not modify or reuse <data> after this call.
        """
        if not isinstance(data, array.array):
            raise TypeError("Data must be an array.array.")

        if data.typecode != 'f' or data.itemsize != 4:
            raise RuntimeError("Array must use 4-byte float32 storage.")

        expected_numel = prod(shape)

        if len(data) != expected_numel:
            raise ValueError(
                f"Expected {expected_numel} elements, "
                f"but received {len(data)}."
            )

        tensor = cls.__new__(cls)
        tensor._data = data
        tensor._shape = shape
        return tensor

    # ==========================================
    # Arithmetic Operations
    # ==========================================

    @check_add_shapes
    def __add__(self, x: Tensor) -> Tensor:
        """
        Performs element-wise addition between this tensor and <x>.

        Inputs:
            x: The second tensor operand. It must have the same shape as this
                tensor, or this tensor must have shape (N, D) and <x> must
                have shape (D,).

        Outputs:
            A new Tensor containing the element-wise sums of this tensor and
            <x>.

        Raises:
            TypeError: If <x> is not a Tensor.
            ValueError: If this tensor and <x> do not have compatible
                shapes.
        """
        # TODO
        raise NotImplementedError

    def __mul__(self, c: float) -> Tensor:
        """
        Performs scalar multiplication between this tensor and <c>.

        Inputs:
            c: The scalar multiplier.

        Outputs:
            A new Tensor with the same shape as this tensor, containing each
            element multiplied by <c>.

        Raises:
            TypeError: If <c> is not a real number.
        """
        # TODO
        raise NotImplementedError

    @check_matmul_shapes
    def __matmul__(self, x: Tensor) -> Tensor:
        """
        Performs matrix multiplication between this tensor and <x>.

        Inputs:
            x: The right tensor operand. Both tensors must be 2D, and the
                number of columns in this tensor must equal the number of rows
                in <x>.

        Outputs:
            A new Tensor containing the matrix product of this tensor and
            <x>.

        Raises:
            TypeError: If <x> is not a Tensor.
            ValueError: If the dimensions of this tensor and <x> are
                incompatible for matrix multiplication.
        """
        # TODO
        raise NotImplementedError

    # ==========================================
    # Metadata Properties
    # ==========================================

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the tensor.

        Equivalent to torch.Tensor.shape

        Inputs:
            None.

        Outputs:
            A tuple of integers containing the size of the tensor in each
            dimension.
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions (rank) of the tensor.

        Equivalent to torch.Tensor.ndim.

        Inputs:
            None.

        Outputs:
            An integer equal to the number of dimensions in the tensor.
        """
        return len(self._shape)

    @property
    def numel(self) -> int:
        """
        Returns the number of elements in the tensor.

        Equivalent to torch.Tensor.numel, exposed here as a property rather
        than a method.

        Inputs:
            None.

        Outputs:
            An integer equal to the total number of elements in the tensor.
        """
        return len(self._data)

    # ==========================================
    # Access and Inspection
    # ==========================================

    def __repr__(self) -> str:
        """
        Returns a string representation of the tensor.

        Inputs:
            None.

        Outputs:
            A string representation of this tensor.
        """
        # TODO
        raise NotImplementedError

    def __getitem__(self, key) -> float | Tensor:
        """
        Reads the value stored in the tensor at location <key>.

        Inputs:
            key: An index or collection of indices identifying a value or
                sub-tensor.

        Outputs:
            A floating-point value when <key> identifies one element, or a
            view of the Tensor when <key> identifies multiple elements.

        Raises:
            IndexError: If <key> is outside the tensor bounds.
            TypeError: If <key> or <value> has an unsupported type.
        """
        # TODO
        raise NotImplementedError

    def __setitem__(self, key, value) -> None:
        """
        Writes values into the tensor at location <key>.

        Inputs:
            key: An index or collection of indices identifying the value or
                sub-tensor to update.
            value: A floating-point value or Tensor containing the replacement
                data.

        Outputs:
            None. Updates this Tensor instance in place.

        Raises:
            IndexError: If <key> is outside the tensor bounds.
            TypeError: If <key> or <value> has an unsupported type.
            ValueError: If <value> has a shape incompatible with the selected
                elements.
        """
        # TODO
        raise NotImplementedError