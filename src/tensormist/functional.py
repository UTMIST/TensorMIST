"""
An implementation of functional operators that support tensors.
"""

import array

from tensormist import _C
from tensormist.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """
    Applies the rectified linear unit (ReLU) function to <x>.

    Each negative element is replaced with zero, while non-negative elements
    are unchanged.

    Inputs:
        x: The input Tensor to which ReLU is applied.

    Outputs:
        A Tensor with the same shape as <x>, containing the element-wise
        ReLU results.
    """
    out = array.array("f", [0.0]) * x.numel
    _C.relu(x._data, out, 1, x.numel)
    return Tensor(out, x.shape)
