import numpy as np

from gradient_descent.gradient_descent import numerical_gradient

# tag::numerical_gradient_eg
f = lambda x: (1 / 16) * x[0] ** 2 + 9 * x[1] ** 2
param = np.array([2.0, 3.0])
numerical_grads = numerical_gradient(lambda: f(param), [param])
print(numerical_grads[0])
# end::numerical_gradient_eg
