import numpy as np

from regression.regression import (gradient_softmax, gradient_softmax_onehot, loss_softmax,
                                   loss_softmax_onehot)

# tag::softmax_gradient_example
X = np.array([[2, 3], [4, 5]])
y = np.array([2, 1])
W = np.array([[0.1, 0.2, 0.3], [0.4, 0.2, 0.8]])
reg = 0.2
print(f"{gradient_softmax(W,X,y,reg) = }")
print(f"{loss_softmax(W,X,y,reg) = }")
# one-hot representation
X = np.array([[2, 3], [4, 5]])
y = np.array([[0, 0, 1], [0, 1, 0]])
print(f"{gradient_softmax_onehot(W,X,y,reg) = }")
print(f"{loss_softmax_onehot(W,X,y,reg) = }")
# end::softmax_gradient_example
