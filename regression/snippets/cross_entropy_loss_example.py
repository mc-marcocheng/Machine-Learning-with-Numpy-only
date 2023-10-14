import numpy as np

from regression.regression import cross_entropy

# tag::cross_entropy_loss_example
F = np.array([[0.2, 0.5, 0.3], [0.2, 0.6, 0.2]])
Y = np.array([2, 1])
print(cross_entropy(F, Y))
# end::cross_entropy_loss_example
