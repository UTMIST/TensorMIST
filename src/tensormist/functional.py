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

def sum(x: Tensor, dim: int = None) -> Tensor:
    """
    Applies sum reduction to <x> or, if <x> is 2D, applies sum reduction to rows or columns of <x> depending on <dim>
    
    Inputs:
        x: The input Tensor whose elements will be summed.
        dim: Optional argument to specify which axis to sum along only if <x> is 2D.

    Outputs:
        If <dim> is None, returns a Tensor with shape (), containing the sum of elements of <x>.
        If <dim> is 0 and <x> has shape (R, C), returns a Tensor with shape (C,), containing the sum of the rows of <x>.
        If <dim> is 1 and <x> has shape (R, C), returns a Tensor with shape (R,), containing the sum of the columns of <x>.

    Raises:
        TypeError: If <x> is not a Tensor.
        ValueError: If <dim>'s value is incorrect based on the shape of <x>.
    """
    raise NotImplementedError

def softmax(x: Tensor) -> Tensor:
    """
    Applies softmax function to <x>

    Inputs:
        x: The input Tensor to which softmax is applied.

    Outputs:
        A Tensor with same shape <x> containing the components of the softmax of <x>.

    Raises:
        TypeError: If <x> is not a Tensor.    
    """
    raise NotImplementedError

def sigmoid(x: Tensor) -> Tensor:
    """
    Applies sigmoid function to elements of <x>.

    Inputs:
        x: The input Tensor to whose elements sigmoid is applied to.

    Outputs:
        A Tensor with same shape <x>, containing element-wise sigmoid results.

    Raises:
        TypeError: If <x> is not a Tensor.    
    """
    raise NotImplementedError

def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """
    Calculates binary cross-entropy loss of the <pred> distribution based on the <target> distribution
    
    Inputs:
        pred: The input Tensor that corresponds to the predicted probabilities
        target: The input Tensor that corresponds to the true, smooth labels

    Outputs:
        A Tensor with shape (), containing the binary cross-entropy loss of <pred> based on <target>

    Raises:
        TypeError: If <pred> or <target> is not a Tensor.
        ValueError: If <pred> and <target> do not have the same shape.
    """
    raise NotImplementedError

def cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
     """
    Calculates cross-entropy loss of the unnormalized <logits> based on the <target> distribution
    
    Inputs:
        logits: The input Tensor that corresponds to the unnormalized logits
        target: The input Tensor that corresponds to the true, smooth labels

    Outputs:
        A Tensor with shape (), containing the cross-entropy loss of <logits> based on <target>

    Raises:
        TypeError: If <logits> or <target> is not a Tensor.
        ValueError: If <logits and <target> do not have the same shape.
    """
    raise NotImplementedError

def tanh(x: Tensor) -> Tensor:
    """
    Applies element-wise the hyperbolic tangent function (tanh) to <x>

    Inputs:
        x: The input Tensor to which tanh is applied

    Outputs:
        A Tensor with the same shape as <x>, containing the outputs of tanh

    Raises:
        TypeError: If <x> is not a Tensor.
    """
    raise NotImplementedError

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Calculates mean squared error of the predictions in <pred> from the ground truth values of <target>

    Inputs:
        pred: The input Tensor containing the predicted values
        target: The input Tensor containing the ground truth values

    Outputs:
        A Tensor with shape () that containing the mean squared error over all the elements of <pred> based on true values in <target>

    Raises:
       TypeError: If <pred> or <target> is not a Tensor.
       ValueError: If <pred> and <target> have different shapes.
    """
    raise NotImplementedError
