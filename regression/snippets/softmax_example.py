import numpy as np

from regression.regression import softmax

# tag::softmax_example
z = np.array([3.0, 1.0, 2.0])
print(f"softmax([3.0, 1.0, 2.0]) =\n{softmax(z)}")

z = np.array([[1, 2, 3], [6, 2, 4]])
print(f"softmax([[1, 2, 3], [6, 2, 4]]) =\n{softmax(z)}")
# end::softmax_example
