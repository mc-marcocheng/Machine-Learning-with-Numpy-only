import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

# tag::multivariate_toy_dataset
n_points = 20
a, b, c = 3, 2, 5
x_range, y_range, noise = 5, 5, 3

xs = np.random.uniform(-x_range, x_range, n_points)
ys = np.random.uniform(-y_range, y_range, n_points)
zs = xs * a + ys * b + np.random.normal(scale=noise)
# end::multivariate_toy_dataset

xx, yy = np.meshgrid([*range(-x_range, x_range + 1)], [*range(-y_range, y_range + 1)])
zz = a * xx + b * yy + c
plt3d = plt.figure().add_subplot(projection="3d")
plt3d.plot_surface(xx, yy, zz, alpha=0.2)

ax = plt.gca()
ax.scatter(xs, ys, zs, color="b")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.savefig("assets/multivariate_plane.png")

# tag::multivariate_toy_dataset_normal_equation
X = np.hstack((np.ones((len(xs), 1), dtype=xs.dtype), xs[:, None], ys[:, None]))
y = zs
w = np.linalg.inv(X.T @ X) @ (X.T @ y)
residual = np.linalg.norm(y - X @ w)
print(f"Resulting plane: z = {w[1]}x + {w[2]}y + {w[0]}")
print(f"Error: {residual}")
# end::multivariate_toy_dataset_normal_equation

xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx2, yy2 = np.meshgrid(np.arange(xlim[0], xlim[1]), np.arange(ylim[0], ylim[1]))
zz2 = w[1] * xx2 + w[1] * yy2 + w[0]
zs2 = w[1] * xs + w[1] * ys + w[0]
plt3d = plt.figure().add_subplot(projection="3d")
plt3d.plot_surface(xx, yy, zz, alpha=0.5)
plt3d.plot_wireframe(xx2, yy2, zz2, color="k", alpha=0.2)

ax = plt.gca()
ax.scatter(xs, ys, zs, color="b")
ax.scatter(xs, ys, zs2, color="r")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.savefig("assets/multivariate_plane_normal_equation.png")
