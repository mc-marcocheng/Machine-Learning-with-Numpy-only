import numpy as np

from regression.regression import softmax_cross_entropy_one_hot

# tag::softmax_cross_entropy_one_hot_example
Z = np.array([[2, 25, 13], [54, 3, 11]])
y = np.array([[0, 0, 1], [0, 1, 0]])
print(softmax_cross_entropy_one_hot(Z, y))
# end::softmax_cross_entropy_one_hot_example
