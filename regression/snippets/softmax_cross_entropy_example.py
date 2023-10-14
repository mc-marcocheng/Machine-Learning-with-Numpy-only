import numpy as np

from regression.regression import softmax_cross_entropy

# tag::softmax_cross_entropy_example
Z = np.array([[2, 25, 13], [54, 3, 11]])
y = np.array([2, 1])
print(softmax_cross_entropy(Z, y))
# end::softmax_cross_entropy_example
