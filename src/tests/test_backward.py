import numpy as np
import pytest

from tensormist import _C

EPS = 1e-3


def rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=shape).astype(np.float32)


def empty(shape):
    return np.full(shape, np.nan, dtype=np.float32)


def add(a, b):
    out = empty(a.shape)
    _C.add(a, b, out, *a.shape)
    return out


def add_backward(a, b, g):
    ga, gb = empty(a.shape), empty(b.shape)
    _C.add_backward(g, ga, gb, *a.shape)
    return ga, gb


def subtract(a, b):
    out = empty(a.shape)
    _C.subtract(a, b, out, *a.shape)
    return out


def subtract_backward(a, b, g):
    ga, gb = empty(a.shape), empty(b.shape)
    _C.subtract_backward(g, ga, gb, *a.shape)
    return ga, gb


def multiply(a, b):
    out = empty(a.shape)
    _C.multiply(a, b, out, *a.shape)
    return out


def multiply_backward(a, b, g):
    ga, gb = empty(a.shape), empty(b.shape)
    _C.multiply_backward(a, b, g, ga, gb, *a.shape)
    return ga, gb


def scalar_multiply(a):
    out = empty(a.shape)
    _C.scalar_multiply(a, -1.5, out, *a.shape)
    return out


def scalar_multiply_backward(a, g):
    ga = empty(a.shape)
    _C.scalar_multiply_backward(g, -1.5, ga, *a.shape)
    return (ga,)


def matmul(a, b):
    m, k = a.shape
    n = b.shape[1]
    out = empty((m, n))
    _C.matmul(a, b, out, m, k, n)
    return out


def matmul_backward(a, b, g):
    m, k = a.shape
    n = b.shape[1]
    ga, gb = empty(a.shape), empty(b.shape)
    _C.matmul_backward(a, b, g, ga, gb, m, k, n)
    return ga, gb


def transpose(a):
    out = empty(a.shape[::-1])
    _C.transpose(a, out, *a.shape)
    return out


def transpose_backward(a, g):
    ga = empty(a.shape)
    _C.transpose_backward(g, ga, *a.shape)
    return (ga,)


def relu(a):
    out = empty(a.shape)
    _C.relu(a, out, *a.shape)
    return out


def relu_backward(a, g):
    ga = empty(a.shape)
    _C.relu_backward(a, g, ga, *a.shape)
    return (ga,)


def bias_add(a, bias):
    out = empty(a.shape)
    _C.bias_add(a, bias, out, *a.shape)
    return out


def bias_add_backward(a, bias, g):
    ga, gbias = empty(a.shape), empty(bias.shape)
    _C.bias_add_backward(g, ga, gbias, *a.shape)
    return ga, gbias


def sum_all(a):
    out = empty((1,))
    _C.sum(a, out, a.size)
    return out


def sum_all_backward(a, g):
    ga = empty(a.shape)
    _C.sum_backward(g, ga, a.size)
    return (ga,)


def sum_rows(a):
    out = empty((a.shape[0],))
    _C.sum_rows(a, out, *a.shape)
    return out


def sum_rows_backward(a, g):
    ga = empty(a.shape)
    _C.sum_rows_backward(g, ga, *a.shape)
    return (ga,)


def sum_cols(a):
    out = empty((a.shape[1],))
    _C.sum_cols(a, out, *a.shape)
    return out


def sum_cols_backward(a, g):
    ga = empty(a.shape)
    _C.sum_cols_backward(g, ga, *a.shape)
    return (ga,)


# sigmoid, tanh and softmax backward take the forward output, not the input.

def sigmoid(a):
    out = empty(a.shape)
    _C.sigmoid(a, out, *a.shape)
    return out


def sigmoid_backward(a, g):
    ga = empty(a.shape)
    _C.sigmoid_backward(sigmoid(a), g, ga, *a.shape)
    return (ga,)


def tanh(a):
    out = empty(a.shape)
    _C.tanh(a, out, *a.shape)
    return out


def tanh_backward(a, g):
    ga = empty(a.shape)
    _C.tanh_backward(tanh(a), g, ga, *a.shape)
    return (ga,)


def softmax(a):
    out = empty(a.shape)
    _C.softmax(a, out, *a.shape)
    return out


def softmax_backward(a, g):
    ga = empty(a.shape)
    _C.softmax_backward(softmax(a), g, ga, *a.shape)
    return (ga,)


# Losses return a (1,) array holding the mean. The kernels give no gradient for
# the target, so backward returns None there and check() skips it.

def mse_loss(pred, target):
    out = empty((1,))
    _C.mse_loss(pred, target, out, *pred.shape)
    return out


def mse_loss_backward(pred, target, g):
    gp = empty(pred.shape)
    _C.mse_loss_backward(pred, target, g, gp, *pred.shape)
    return gp, None


def bce_loss(pred, target):
    out = empty((1,))
    _C.binary_cross_entropy(pred, target, out, *pred.shape)
    return out


def bce_loss_backward(pred, target, g):
    gp = empty(pred.shape)
    _C.binary_cross_entropy_backward(pred, target, g, gp, *pred.shape)
    return gp, None


def cross_entropy(logits, labels):
    out = empty((1,))
    _C.cross_entropy(logits, labels, out, *logits.shape)
    return out


def cross_entropy_backward(logits, labels, g):
    gl = empty(logits.shape)
    _C.cross_entropy_backward(logits, labels, g, gl, *logits.shape)
    return gl, None


A = rand((2, 4), seed=1)
B = rand((2, 4), seed=2)
C = rand((4, 3), seed=3)
BIAS = rand((4,), seed=4)
RELU_X = np.array([[-0.9, -0.4, 0.3, 0.8], [0.5, -0.7, -0.2, 0.6]], dtype=np.float32)
PROBS = np.random.default_rng(6).uniform(0.1, 0.9, size=(2, 4)).astype(np.float32)  # bce preds in (0, 1)
LABELS_01 = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.float32)
CLASS_LABELS = np.array([1, 3], dtype=np.uintp)  # one class index per row, stored as size_t

CASES = [
    pytest.param(add, add_backward, (A, B), id="add"),
    pytest.param(subtract, subtract_backward, (A, B), id="subtract"),
    pytest.param(multiply, multiply_backward, (A, B), id="multiply"),
    pytest.param(scalar_multiply, scalar_multiply_backward, (A,), id="scalar_multiply"),
    pytest.param(matmul, matmul_backward, (A, C), id="matmul"),
    pytest.param(transpose, transpose_backward, (A,), id="transpose"),
    pytest.param(relu, relu_backward, (RELU_X,), id="relu"),
    pytest.param(bias_add, bias_add_backward, (A, BIAS), id="bias_add"),
    pytest.param(sum_all, sum_all_backward, (A,), id="sum"),
    pytest.param(sum_rows, sum_rows_backward, (A,), id="sum_rows"),
    pytest.param(sum_cols, sum_cols_backward, (A,), id="sum_cols"),
    pytest.param(sigmoid, sigmoid_backward, (A,), id="sigmoid"),
    pytest.param(tanh, tanh_backward, (A,), id="tanh"),
    pytest.param(softmax, softmax_backward, (A,), id="softmax"),
    pytest.param(mse_loss, mse_loss_backward, (A, B), id="mse"),
    pytest.param(bce_loss, bce_loss_backward, (PROBS, LABELS_01), id="bce"),
    pytest.param(cross_entropy, cross_entropy_backward, (A, CLASS_LABELS), id="cross_entropy"),
]


def check(forward, backward, inputs, loss, grad_out):
    """Compares backward's gradients with finite differences of loss(forward(*inputs))."""
    grads = backward(*inputs, grad_out)

    for j, x in enumerate(inputs):
        if grads[j] is None:  # no gradient for this input (e.g. a loss target)
            continue
        for i in range(x.size):
            plus = [t.copy() for t in inputs]
            minus = [t.copy() for t in inputs]
            plus[j].flat[i] += EPS
            minus[j].flat[i] -= EPS

            numerical = (loss(forward(*plus)) - loss(forward(*minus))) / (2 * EPS)
            assert grads[j].flat[i] == pytest.approx(numerical, rel=1e-2, abs=1e-3), f"input {j}, element {i}"


def sum_loss(out):
    return out.sum()


def square_loss(out):
    return (out * out).sum() / 2


@pytest.mark.parametrize("forward, backward, inputs", CASES)
class TestBackward:

    def test_trivial_loss(self, forward, backward, inputs):
        out = forward(*inputs)
        check(forward, backward, inputs, loss=sum_loss, grad_out=np.ones_like(out))

    def test_linear_loss(self, forward, backward, inputs):
        alpha = rand(forward(*inputs).shape, seed=5)

        def linear_loss(out):
            return (alpha * out).sum()

        check(forward, backward, inputs, loss=linear_loss, grad_out=alpha)

    def test_quadratic_loss(self, forward, backward, inputs):
        out = forward(*inputs)
        check(forward, backward, inputs, loss=square_loss, grad_out=out)

    def test_mse_loss(self, forward, backward, inputs):
        # e2e: the op's output feeds into the mse kernel, and mse's backward gives grad_out
        out = forward(*inputs).reshape(1, -1)
        target = rand(out.shape, seed=6)

        def loss(o):
            return mse_loss(o.reshape(1, -1), target)[0]

        grad_out, _ = mse_loss_backward(out, target, np.ones(1, dtype=np.float32))
        check(forward, backward, inputs, loss=loss, grad_out=grad_out.reshape(forward(*inputs).shape))
