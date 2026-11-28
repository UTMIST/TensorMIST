"""
An implementation of a tensor object and its supported operations.
"""

from __future__ import annotations

import array
from collections.abc import Iterable
from itertools import product
from math import prod

from tensormist.validation import check_add_shapes, check_matmul_shapes, validates_shape

from tensormist import _C


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

        if isinstance(data, array.array) and (
            data.typecode != "f" or data.itemsize != 4
        ):
            raise RuntimeError("Array must use 4-byte float32 storage.")

        try:
            self._data = array.array("f", data)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("Data must contain valid numeric values.") from exc

        # Validation: Check the float32 element size and ensure the number
        # of stored elements matches the product of the shape dimensions.
        if self._data.itemsize != 4:
            raise RuntimeError("Float32 must occupy 4 bytes.")

        expected_numel = prod(shape)

        if len(self._data) != expected_numel:
            raise ValueError(
                f"Expected {expected_numel} elements, but received {len(self._data)}."
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
        return cls._from_array(array.array("f", [1.0]) * prod(shape), shape)

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
        return cls._from_array(array.array("f", [0.0]) * prod(shape), shape)

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

        if data.typecode != "f" or data.itemsize != 4:
            raise RuntimeError("Array must use 4-byte float32 storage.")

        expected_numel = prod(shape)

        if len(data) != expected_numel:
            raise ValueError(
                f"Expected {expected_numel} elements, but received {len(data)}."
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
        out = Tensor.zeros(self.shape)
        
        if self.shape == x.shape:
            _C.add(self._data, x._data, out._data, 1, self.numel)
        else:
            _C.bias_add(self._data, x._data, out._data, self.shape[0], self.shape[1])
                        
        return out
        

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
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise TypeError("First operand is not a real number.")
        
        out = Tensor.zeros(self.shape)
        
        _C.scalar_multiply(self._data, c, out._data, 1, self.numel)

        return out

    __rmul__ = __mul__

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
        out = Tensor.zeros((self.shape[0], x.shape[1]))
        
        _C.matmul(self._data, x._data, out._data, self.shape[0], self.shape[1], x.shape[1])

        return out
        

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

        def build_repr(shape):
            nonlocal offset

            if not shape:
                value = self._data[offset]
                offset += 1
                return value

            values = []
            for _ in range(shape[0]):
                values.append(build_repr(shape[1:]))

            return values

        offset = 0
        nested_repr = build_repr(self._shape)
        return f"{type(self).__name__}({nested_repr!r})"

    def _flat_indices(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> tuple[list[int], tuple[int, ...]]:
        if isinstance(key, tuple):
            specs = key
        elif isinstance(key, (int, slice)) and not isinstance(key, bool):
            specs = (key,)
        else:
            raise TypeError("Invalid tensor index.")

        if len(specs) > self.ndim:
            raise IndexError("Too many tensor indices.")

        specs += (slice(None),) * (self.ndim - len(specs))
        selected_axes = []
        selected_shape = []

        for size, dim in zip(self._shape, specs):
            if isinstance(dim, int) and not isinstance(dim, bool):
                index = dim + size if dim < 0 else dim
                if index < 0 or index >= size:
                    raise IndexError("Tensor index out of bounds.")
                selected_axes.append((index,))
            elif isinstance(dim, slice):
                slice_indices = range(*dim.indices(size))
                selected_axes.append(slice_indices)
                selected_shape.append(len(slice_indices))
            else:
                raise TypeError("Invalid tensor index.")

        selected_indices = []
        for coords in product(*selected_axes):
            curr = 0
            for c, size in zip(coords, self._shape):
                curr = curr * size + c
            selected_indices.append(curr)

        return selected_indices, tuple(selected_shape)

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> float | Tensor:
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
            TypeError: If <key> has an unsupported type.
        """
        selected_indices, selected_shape = self._flat_indices(key)
        if not selected_shape:
            return self._data[selected_indices[0]]

        return Tensor((self._data[index] for index in selected_indices), selected_shape)

    def __setitem__(
        self, key: int | slice | tuple[int | slice, ...], value: float | Tensor
    ) -> None:
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
        selected_indices, selected_shape = self._flat_indices(key)

        if isinstance(value, Tensor):
            if value.shape != selected_shape:
                raise ValueError(
                    f"Expected a value with shape {selected_shape}, got {value.shape}."
                )
            values = value._data
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values = [value] * len(selected_indices)
        else:
            raise TypeError("Value must be a number or a Tensor.")

        for index, replacement in zip(selected_indices, values):
            self._data[index] = replacement
