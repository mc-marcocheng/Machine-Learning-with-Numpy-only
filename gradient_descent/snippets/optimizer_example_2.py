import numpy as np

from gradient_descent.gradient_descent import SGD_Momentum, gradient_descent_

df = lambda x: np.array((1 / 8 * x[0], 18 * x[1]))
# tag::gradient_descent_general_SGD_momentum
x0 = np.array([-2.4, 0.2])
optimizer = SGD_Momentum([x0], 0.1, 0.8)
path = gradient_descent_(df, optimizer, 1000)
print(path[-1])
# end::gradient_descent_general_SGD_momentum
