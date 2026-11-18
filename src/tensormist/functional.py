"""
An implementation of functional operators that support tensors.
"""

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
    # TODO
    raise NotImplementedError
