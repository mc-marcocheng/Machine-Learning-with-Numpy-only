import numpy as np

from regression.regression import cross_entropy_one_hot

# tag::cross_entropy_one_hot_example
F = np.array([[0.2, 0.5, 0.3], [0.2, 0.6, 0.2]])
Y = np.array([[0, 0, 1], [0, 1, 0]])
print(cross_entropy_one_hot(F, Y))
# end::cross_entropy_one_hot_example
