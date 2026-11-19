"""Smoke checks for the installed TensorMIST package.

Run from outside the checkout after installing the built wheel:
    python /path/to/TensorMIST/src/tests/test_package.py
"""

from array import array
import importlib
import unittest


class PackageSmokeTests(unittest.TestCase):
    def test_imports(self):
        # Check that the package and its parts can be imported 
        for name in ("tensormist", "tensormist.functional", "tensormist._C"):
            with self.subTest(module=name):
                importlib.import_module(name)

        # Check that the package exports the same Tensor class defined in tensor.py
        from tensormist import Tensor
        from tensormist.tensor import Tensor as TensorClass

        self.assertIs(Tensor, TensorClass)

    # Check that python can call the C backend
    def test_native_add(self):
        from tensormist import _C

        a = array("f", [1, -2, 0, 4, 5, -6])
        b = array("f", [3, 2, -1, -4, 2, 1])
        out = array("f", [7] * 6)

        _C.add(a, b, out, 2, 3)

        self.assertEqual(list(out), [4, 0, -1, 0, 7, -5])


if __name__ == "__main__":
    unittest.main()
