import numpy as np

from gradient_descent.gradient_descent import SGD, gradient_descent_

# tag::gradient_descent_general_SGD
df = lambda x: np.array((1 / 8 * x[0], 18 * x[1]))
x0 = np.array([-2.4, 0.2])
optimizer = SGD([x0], 0.1)
path = gradient_descent_(df, optimizer, 100)
print(path[-1])
# end::gradient_descent_general_SGD
