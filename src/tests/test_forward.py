import numpy as np
import torch
import pytest

from tensormist import _C

def test_add_forward():
    
    a = np.array(
        [[1, -2, 3],
         [4, 5, -6]],
        dtype=np.float32
    )

    b = np.array(
        [[7, 8, -9],
         [10, -11, 12]],
        dtype=np.float32
    )

    
    a_before = a.copy()
    b_before = b.copy()

    
    expected = (
        torch.from_numpy(a.copy())
        + torch.from_numpy(b.copy())
    ).numpy()

    
    out = np.full((2, 3), 12345.0, dtype=np.float32)

    
    _C.add(a, b, out, 2, 3)

    np.testing.assert_allclose(
        out, expected, rtol=1e-5, atol=1e-6
    )

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


@pytest.mark.parametrize("op", [
    "add",
    "subtract",
    "multiply",
])
@pytest.mark.parametrize("shape", [
    (1, 1),
    (2, 3),
    (3, 2),
    (8, 17),
])

def test_elementwise_forward(op, shape):
    rng = np.random.default_rng(42)

    a = np.ascontiguousarray(
        rng.normal(size=shape),
        dtype=np.float32
    )

    b = np.ascontiguousarray(
        rng.normal(size=shape),
        dtype=np.float32
    )

    a_before = a.copy()
    b_before = b.copy()

    torch_a = torch.from_numpy(a.copy())
    torch_b = torch.from_numpy(b.copy())

    if op == "add":
        expected = torch_a + torch_b
    elif op == "subtract":
        expected = torch_a - torch_b
    else:
        expected = torch_a * torch_b

    out = np.full(shape, 12345.0, dtype=np.float32)

    kernel = getattr(_C, op)
    kernel(a, b, out, *shape)

    np.testing.assert_allclose(
        out,
        expected.numpy(),
        rtol=1e-5,
        atol=1e-6,
    )

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


@pytest.mark.parametrize("shape, scalar", [
    ((1, 1), 2.0),
    ((2, 3), -1.5),
    ((3, 2), 0.0),
    ((4, 5), 3.25),
    ((8, 17), 1.0),
])
def test_scalar_multiply_forward(shape, scalar):
    rng = np.random.default_rng(43)

    a = np.ascontiguousarray(
        rng.normal(size=shape),
        dtype=np.float32
    )

    a_before = a.copy()

    expected = (
        torch.from_numpy(a.copy()) * scalar
    ).numpy()

    out = np.full(shape, 12345.0, dtype=np.float32)

    _C.scalar_multiply(a, scalar, out, *shape)

    np.testing.assert_allclose(
        out,
        expected,
        rtol=1e-5,
        atol=1e-6
    )

    np.testing.assert_array_equal(a, a_before)

@pytest.mark.parametrize("m, k, n", [
    (1, 1, 1),
    (2, 3, 4),
    (4, 3, 2),
    (5, 1, 3),
    (3, 7, 5),
])
def test_matmul_forward(m, k, n):
    rng = np.random.default_rng(44)

    a = np.ascontiguousarray(
        rng.normal(size=(m, k)),
        dtype=np.float32
    )

    b = np.ascontiguousarray(
        rng.normal(size=(k, n)),
        dtype=np.float32
    )

    a_before = a.copy()
    b_before = b.copy()

    expected = (
        torch.from_numpy(a.copy())
        @ torch.from_numpy(b.copy())
    ).numpy()

    out = np.full(
        (m, n),
        12345.0,
        dtype=np.float32
    )

    _C.matmul(a, b, out, m, k, n)

    assert out.shape == (m, n)

    np.testing.assert_allclose(
        out,
        expected,
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)

@pytest.mark.parametrize("rows, cols", [
    (1, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
    (5, 7),
])
def test_transpose_forward(rows, cols):
    rng = np.random.default_rng(45)

    a = np.ascontiguousarray(
        rng.normal(size=(rows, cols)),
        dtype=np.float32
    )

    a_before = a.copy()

    expected = (
        torch.from_numpy(a.copy()).T
    ).numpy()

    out = np.full(
        (cols, rows),
        12345.0,
        dtype=np.float32
    )

    _C.transpose(a, out, rows, cols)

    assert out.shape == (cols, rows)

    np.testing.assert_allclose(
        out,
        expected,
        rtol=1e-5,
        atol=1e-6
    )

    np.testing.assert_array_equal(a, a_before)


@pytest.mark.parametrize("shape", [
    (1, 1),
    (2, 3),
    (3, 2),
    (8, 17),
])
def test_relu_forward(shape):
    rng = np.random.default_rng(46)

    a = np.ascontiguousarray(
        rng.normal(size=shape),
        dtype=np.float32
    )
    a_before = a.copy()

    expected = torch.relu(torch.from_numpy(a.copy())).numpy()
    out = np.full(shape, 12345.0, dtype=np.float32)

    _C.relu(a, out, *shape)

    np.testing.assert_allclose(
        out, expected, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(a, a_before)

def test_relu_boundary_values():
    a = np.ascontiguousarray(
        [[-100.0, -1e-6, 0.0, 1e-6, 100.0]],
        dtype=np.float32
    )

    expected = torch.relu(torch.from_numpy(a.copy())).numpy()
    out = np.full(a.shape, 12345.0, dtype=np.float32)

    _C.relu(a, out, 1, 5)

    np.testing.assert_array_equal(out, expected)

@pytest.mark.parametrize("m, n", [
    (1, 1),
    (1, 5),
    (2, 3),
    (4, 1),
    (5, 7),
])
def test_bias_add_forward(m, n):
    rng = np.random.default_rng(47)

    a = np.ascontiguousarray(
        rng.normal(size=(m, n)),
        dtype=np.float32
    )

    bias = np.ascontiguousarray(
        rng.normal(size=(n,)),
        dtype=np.float32
    )

    a_before = a.copy()
    bias_before = bias.copy()

    expected = (
        torch.from_numpy(a.copy())
        + torch.from_numpy(bias.copy())
    ).numpy()

    out = np.full((m, n), 12345.0, dtype=np.float32)

    _C.bias_add(a, bias, out, m, n)

    np.testing.assert_allclose(
        out,
        expected,
        rtol=1e-5,
        atol=1e-6
    )

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(bias, bias_before)

@pytest.mark.parametrize("n", [1, 2, 5, 16, 100])
def test_sum_forward(n):
    rng = np.random.default_rng(48)

    a = np.ascontiguousarray(
        rng.normal(size=n),
        dtype=np.float32
    )
    a_before = a.copy()

    expected = torch.from_numpy(a.copy()).sum().item()

    out = np.full((1,), 12345.0, dtype=np.float32)

    _C.sum(a, out, n)

    np.testing.assert_allclose(
        out[0], expected, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(a, a_before)

@pytest.mark.parametrize("m, n", [
    (1, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
    (8, 17),
])
def test_sum_rows_forward(m, n):
    rng = np.random.default_rng(49)

    a = np.ascontiguousarray(
        rng.normal(size=(m, n)),
        dtype=np.float32
    )
    a_before = a.copy()

    expected = (
        torch.from_numpy(a.copy())
        .sum(dim=1)
        .numpy()
    )

    out = np.full((m,), 12345.0, dtype=np.float32)

    _C.sum_rows(a, out, m, n)

    np.testing.assert_allclose(
        out, expected, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(a, a_before)

@pytest.mark.parametrize("m, n", [
    (1, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
    (8, 17),
])
def test_sum_cols_forward(m, n):
    rng = np.random.default_rng(50)

    a = np.ascontiguousarray(
        rng.normal(size=(m, n)),
        dtype=np.float32
    )
    a_before = a.copy()

    expected = (
        torch.from_numpy(a.copy())
        .sum(dim=0)
        .numpy()
    )

    out = np.full((n,), 12345.0, dtype=np.float32)

    _C.sum_cols(a, out, m, n)

    np.testing.assert_allclose(
        out, expected, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(a, a_before)

@pytest.mark.parametrize("op", [
    "sum",
    "sum_rows",
    "sum_cols",
])
def test_reduction_cancellation(op):
    a = np.array([
        [10, -10,  5],
        [ 3,  -3, -5],
    ], dtype=np.float32)

    a_before = a.copy()
    torch_a = torch.from_numpy(a.copy())

    if op == "sum":
        expected = torch_a.sum().numpy()
        out = np.full((1,), 12345.0, dtype=np.float32)
        _C.sum(a, out, a.size)
        actual = out[0]

    elif op == "sum_rows":
        expected = torch_a.sum(dim=1).numpy()
        out = np.full((2,), 12345.0, dtype=np.float32)
        _C.sum_rows(a, out, 2, 3)
        actual = out

    else:
        expected = torch_a.sum(dim=0).numpy()
        out = np.full((3,), 12345.0, dtype=np.float32)
        _C.sum_cols(a, out, 2, 3)
        actual = out

    np.testing.assert_allclose(
        actual, expected, rtol=1e-5, atol=1e-6
    )

    np.testing.assert_array_equal(a, a_before)