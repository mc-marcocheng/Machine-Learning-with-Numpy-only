import numpy as np

from gradient_descent.gradient_descent import numerical_gradient
from regression.regression import grad_softmax_cross_entropy, softmax_cross_entropy

# tag::softmax_cross_entropy_gradient_example
Z = np.array([[2.0, 25.0, 13.0], [54.0, 3.0, 11.0]])
y = np.array([2, 1])
print("grad_softmax_cross_entropy:\n", grad_softmax_cross_entropy(Z, y))
# end::softmax_cross_entropy_gradient_example


# tag::softmax_cross_entropy_gradient_validation
def loss_f():
    return softmax_cross_entropy(Z, y)


print("numerical_gradient:\n", numerical_gradient(loss_f, [Z]))
# end::softmax_cross_entropy_gradient_validation
