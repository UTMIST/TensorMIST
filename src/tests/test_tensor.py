import itertools

import pytest

from tensormist import Tensor


def values(t: Tensor) -> list[float]:
    return [t[i] for i in itertools.product(*(range(d) for d in t.shape))]


class TestTensor:
    def test_init(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.numel == 6
        assert values(t) == pytest.approx([1, 2, 3, 4, 5, 6])

    def test_data_shape_mismatch(self):
        with pytest.raises(ValueError):
            Tensor([0.0, 1.0, 2.0], (10, 20))

    def test_ones(self):
        t = Tensor.ones((4, 5))
        assert t.shape == (4, 5)
        assert all(v == 1.0 for v in values(t))

    def test_zeros(self):
        t = Tensor.zeros((4, 5))
        assert t.shape == (4, 5)
        assert all(v == 0.0 for v in values(t))

    def test_getitem_setitem(self):
        t = Tensor.zeros((2, 3))
        t[(1, 2)] = 5.0
        assert t[(1, 2)] == pytest.approx(5.0)
        with pytest.raises(IndexError):
            t[(2, 0)]

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (([1.0, 2.0], (2,)), ([3.0, 4.0], (2,)), [4, 6]),
            (([1.0, 2.0, 3.0, 4.0], (2, 2)), ([10.0, 20.0], (2,)), [11, 22, 13, 24]),
        ],
    )
    def test_add(self, a, b, expected):
        assert values(Tensor(*a) + Tensor(*b)) == pytest.approx(expected)

    def test_add_bad_shape(self):
        with pytest.raises(ValueError):
            Tensor.zeros((3,)) + Tensor.zeros((4,))

    def test_mul(self):
        t = Tensor([1.0, -2.0, 3.0], (3,))
        assert values(t * 2.0) == pytest.approx([2, -4, 6])

    def test_matmul(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
        b = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2))
        out = a @ b
        assert out.shape == (2, 2)
        assert values(out) == pytest.approx([58, 64, 139, 154])

    def test_matmul_bad_shape(self):
        with pytest.raises(ValueError):
            Tensor.zeros((2, 3)) @ Tensor.zeros((2, 3))
