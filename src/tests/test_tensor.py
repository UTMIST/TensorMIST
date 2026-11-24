import array
import itertools

import pytest

from tensormist import Tensor

# Values computed by the C kernels are covered in test_forward.py / test_backward.py.
# These tests only cover the Python layer: storage, shapes, and validation.


def values(t: Tensor) -> list[float]:
    return [t[i] for i in itertools.product(*(range(d) for d in t.shape))]


class TestInit:
    def test_init(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.numel == 6
        assert values(t) == pytest.approx([1, 2, 3, 4, 5, 6])

    def test_scalar(self):
        # shape () is a scalar: zero dimensions, one element
        t = Tensor([5.0], ())
        assert t.shape == ()
        assert t.ndim == 0
        assert t.numel == 1
        assert t[()] == pytest.approx(5.0)

    def test_zero_size(self):
        # a 0 dimension is valid and holds no elements
        t = Tensor([], (2, 0))
        assert t.shape == (2, 0)
        assert t.numel == 0

    def test_ones(self):
        t = Tensor.ones((4, 5))
        assert t.shape == (4, 5)
        assert t.numel == 20
        assert all(v == 1.0 for v in values(t))

    def test_zeros(self):
        t = Tensor.zeros((4, 5))
        assert t.shape == (4, 5)
        assert t.numel == 20
        assert all(v == 0.0 for v in values(t))

    @pytest.mark.parametrize("data, shape", [([0.0, 1.0, 2.0], (10, 20)), ([0.0, 1.0], (3,)), ([], (1,)), ([], ()), ([1.0, 2.0], ())])
    def test_data_shape_mismatch(self, data, shape):
        with pytest.raises(ValueError):
            Tensor(data, shape)

    @pytest.mark.parametrize("shape", [(-1,), (2, -3)])
    def test_negative_dim(self, shape):
        with pytest.raises(ValueError):
            Tensor([], shape)

    @pytest.mark.parametrize("shape", [[2, 3], 6, (2.0, 3), (True, 6), None])
    def test_bad_shape_type(self, shape):
        with pytest.raises(TypeError):
            Tensor([0.0] * 6, shape)

    @pytest.mark.parametrize("data", [None, 1.0, "ab", [1.0, "a"]])
    def test_bad_data_type(self, data):
        with pytest.raises(TypeError):
            Tensor(data, (2,))

    def test_wrong_array_typecode(self):
        with pytest.raises(RuntimeError) as exc:
            Tensor(array.array("d", [1.0, 2.0]), (2,))
        assert exc.type is not NotImplementedError  # NotImplementedError subclasses RuntimeError

    @pytest.mark.parametrize("factory", [Tensor.ones, Tensor.zeros])
    def test_factory_bad_shape(self, factory):
        with pytest.raises(ValueError):
            factory((-1,))
        with pytest.raises(TypeError):
            factory([2, 3])


class TestIndexing:
    def test_getitem_setitem(self):
        t = Tensor.zeros((2, 3))
        t[(1, 2)] = 5.0
        assert t[(1, 2)] == pytest.approx(5.0)

    @pytest.mark.parametrize("idx", [(2, 0), (0, 3)])
    def test_out_of_bounds(self, idx):
        t = Tensor.zeros((2, 3))
        with pytest.raises(IndexError):
            t[idx]
        with pytest.raises(IndexError):
            t[idx] = 1.0

    def test_bad_key(self):
        with pytest.raises(TypeError):
            Tensor.zeros((2, 3))["a"]

    def test_bad_value(self):
        t = Tensor.zeros((2, 3))
        with pytest.raises(TypeError):
            t[(0, 0)] = "a"


class TestOps:
    @pytest.mark.parametrize("a_shape, b_shape", [((2, 3), (2, 3)), ((2, 3), (3,))])
    def test_add_shape(self, a_shape, b_shape):
        out = Tensor.zeros(a_shape) + Tensor.zeros(b_shape)
        assert out.shape == a_shape
        assert out.numel == Tensor.zeros(a_shape).numel

    @pytest.mark.parametrize("a_shape, b_shape", [((3,), (4,)), ((2, 3), (2,)), ((2, 3), (3, 2)), ((3,), (2, 3))])
    def test_add_bad_shape(self, a_shape, b_shape):
        with pytest.raises(ValueError):
            Tensor.zeros(a_shape) + Tensor.zeros(b_shape)

    @pytest.mark.parametrize("other", [1.0, [1.0, 2.0], None])
    def test_add_non_tensor(self, other):
        with pytest.raises(TypeError):
            Tensor.zeros((2,)) + other

    def test_mul_shape(self):
        out = Tensor.zeros((2, 3)) * 2.0
        assert out.shape == (2, 3)
        assert out.numel == 6

    @pytest.mark.parametrize("c", ["2", None, [2.0]])
    def test_mul_non_real(self, c):
        with pytest.raises(TypeError):
            Tensor.zeros((2,)) * c

    @pytest.mark.parametrize("a_shape, b_shape, out_shape", [((2, 3), (3, 4), (2, 4)), ((1, 5), (5, 1), (1, 1))])
    def test_matmul_shape(self, a_shape, b_shape, out_shape):
        out = Tensor.zeros(a_shape) @ Tensor.zeros(b_shape)
        assert out.shape == out_shape
        assert out.numel == out_shape[0] * out_shape[1]

    @pytest.mark.parametrize("a_shape, b_shape", [((2, 3), (2, 3)), ((3,), (3, 2)), ((2, 3), (3,)), ((2, 2, 2), (2, 2))])
    def test_matmul_bad_shape(self, a_shape, b_shape):
        with pytest.raises(ValueError):
            Tensor.zeros(a_shape) @ Tensor.zeros(b_shape)

    @pytest.mark.parametrize("other", [1.0, [[1.0]], None])
    def test_matmul_non_tensor(self, other):
        with pytest.raises(TypeError):
            Tensor.zeros((2, 2)) @ other
