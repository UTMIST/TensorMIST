"""
Gradient checks for backward kernels using finite gradient difference checking.

    forward(x) -> Tensor
    backward(x, grad_out) -> Tensor with the same shape as x
"""

import random

import pytest

from tensormist import Tensor

EPS = 1e-3

CASES = [
   # add cases when tensor backwards are implemented.
]


@pytest.mark.parametrize("forward, backward, x", CASES)
def test_backward(forward, backward, x):
    out = forward(x)
    rng = random.Random(0)
    alpha = Tensor([rng.uniform(-1, 1) for _ in range(out.numel)], out.shape)
    grad = backward(x, alpha)

    def L(t: Tensor) -> float:
        """dummy loss function"""
        return (alpha * forward(t)).sum()

    for i in range(x.numel):
        e = Tensor([float(j == i) for j in range(x.numel)], x.shape)  # 1 at position i
        numerical = (L(x + e * EPS) - L(x - e * EPS)) / (2 * EPS)
        assert (grad * e).sum() == pytest.approx(numerical, rel=1e-2, abs=1e-3)
