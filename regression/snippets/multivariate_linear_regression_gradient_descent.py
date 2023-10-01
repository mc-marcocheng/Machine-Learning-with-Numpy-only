import matplotlib.pyplot as plt
import numpy as np

from regression.regression import compute_loss_history, linear_regression_vec

np.random.seed(1)

n_points = 20
a, b, c = 3, 2, 5
x_range, y_range, noise = 5, 5, 3

xs = np.random.uniform(-x_range, x_range, n_points)
ys = np.random.uniform(-y_range, y_range, n_points)
zs = xs * a + ys * b + np.random.normal(scale=noise)

# tag::multivariate_linear_regression_gradient_descent
X = np.hstack((xs[:, None], ys[:, None]))
y = zs
history = linear_regression_vec(X, y, alpha=0.02, num_iters=100)
print("w:", history[-1])
# end::multivariate_linear_regression_gradient_descent

loss_history = compute_loss_history(X, y, history)
plt.plot(loss_history, linewidth=2)
plt.title("Gradient descent with learning rate = 0.02", fontsize=16)
plt.xlabel("number of iterations", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.grid()
plt.savefig("assets/multivariate_plane_loss_history.png")
