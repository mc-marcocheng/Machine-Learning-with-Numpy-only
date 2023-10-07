import numpy as np

from regression.regression import softmax_backward, softmax_gradient

# tag::softmax_backward_example
z = np.array([[1, 2]])
print(f"gradient of softmax(z) =\n{softmax_gradient(z)}")
df = np.array([1, 3])  # gradient of L w.r.t. f
print(f"dL/dz =\n{softmax_backward(z, df)}")
# end::softmax_backward_example
