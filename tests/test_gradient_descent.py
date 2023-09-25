import pytest
import numpy as np

from gradient_descent.gradient_descent import gradient_descent


class TestGradientDescent:
    def test_gradient_descent(self):
        df = lambda x: 3*x**2 - 6*x - 9
        path = gradient_descent(df, 1, iterations=200)
        assert np.allclose([path[-1]], 3)
