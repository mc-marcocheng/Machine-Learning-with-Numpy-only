import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# tag::forward_propagation
X = np.array([[1.0, 2.0], [3.0, 4.0]])
W1 = np.array([[0.1, 0.3, 0.5, 0.2], [0.4, 0.6, 0.7, 0.1]])
b1 = np.array([0.1, 0.2, 0.3, 0.4])

print(f"{X.shape = }")
print(f"{W1.shape = }")
print(f"{b1.shape = }")

# First layer
Z1 = np.dot(X, W1) + b1
A1 = sigmoid(Z1)
print(f"{Z1 = }")
print(f"{A1 = }")

W2 = np.array([[0.1, 1.4, 0.2], [2.5, 0.6, 0.3], [1.1, 0.7, 0.8], [0.3, 1.5, 2.1]])
b2 = np.array([0.1, 2, 0.3])
print(f"{A1.shape = }")
print(f"{W2.shape = }")
print(f"{b2.shape = }")

# Second layer
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)
print(f"{Z2 = }")
print(f"{A2 = }")
# end::forward_propagation
