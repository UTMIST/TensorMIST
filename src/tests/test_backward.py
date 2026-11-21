
import random

import pytest

from tensormist import Tensor

EPS = 1e-3

CASES = [
    # pytest.param(_C.relu_forward, _C.relu_backward, Tensor([...], (2, 3)), id="relu"),
]


def random_like(t: Tensor, seed: int = 0) -> Tensor:
    rng = random.Random(seed)
    return Tensor([rng.uniform(-1, 1) for _ in range(t.numel)], t.shape)


@pytest.fixture
def target(forward, x) -> Tensor:
    """Real-valued ground truth shaped like forward(x), e.g. for MSE."""
    return random_like(forward(x), seed=1)


@pytest.fixture
def binary_target(forward, x) -> Tensor:
    out = forward(x)
    rng = random.Random(2)
    return Tensor([float(rng.random() < 0.5) for _ in range(out.numel)], out.shape)


def assert_grad_matches(L, grad: Tensor, x: Tensor) -> None:
    for i in range(x.numel):
        e = Tensor([float(j == i) for j in range(x.numel)], x.shape)  # 1 at position i
        numerical = (L(x + e * EPS) - L(x - e * EPS)) / (2 * EPS)
        assert (grad * e).sum() == pytest.approx(numerical, rel=1e-2, abs=1e-3), f"partial {i}"


@pytest.mark.parametrize("forward, backward, x", CASES)
class TestBackward:

    def test_trivial_loss(self, forward, backward, x):

        out = forward(x)
        grad = backward(x, Tensor.ones(out.shape))
        assert_grad_matches(lambda t: forward(t).sum(), grad, x)

    def test_linear_loss(self, forward, backward, x):

        alpha = random_like(forward(x))
        grad = backward(x, alpha)
        assert_grad_matches(lambda t: (alpha * forward(t)).sum(), grad, x)

    def test_quadratic_loss(self, forward, backward, x):

        out = forward(x)
        grad = backward(x, out)
        assert_grad_matches(lambda t: (forward(t) * forward(t)).sum() / 2, grad, x)

    #TODO add loss kernels when ready

