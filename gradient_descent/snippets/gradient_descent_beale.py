import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

from gradient_descent.gradient_descent import gradient_descent, plot_graph

# Beale's function
f = (
    lambda x, y: (1.5 - x + x * y) ** 2
    + (2.25 - x + x * y**2) ** 2
    + (2.625 - x + x * y**3) ** 2
)
minima_ = np.array([[3], [0.5]])
xmin, xmax, xstep = -4.5, 4.5, 0.2
ymin, ymax, ystep = -4.5, 4.5, 0.2
x_list = np.arange(xmin, xmax + xstep, xstep)
y_list = np.arange(ymin, ymax + ystep, ystep)
x, y = np.meshgrid(x_list, y_list)
z = f(x, y)
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d", elev=50, azim=-50)
ax.plot_surface(
    x,
    y,
    z,
    norm=LogNorm(),
    rstride=1,
    cstride=1,
    edgecolor="none",
    alpha=0.8,
    cmap=plt.cm.jet,
)
ax.plot(*minima_, f(*minima_), "r*", markersize=15)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.savefig("assets/beale_function.png")

# tag::beale_function_gradient_descent
df = lambda x: np.array(
    [
        2 * (1.5 - x[0] + x[0] * x[1]) * (x[1] - 1)
        + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (x[1] ** 2 - 1)
        + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (x[1] ** 3 - 1),
        2 * (1.5 - x[0] + x[0] * x[1]) * x[0]
        + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (2 * x[0] * x[1])
        + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (3 * x[0] * x[1] ** 2),
    ]
)
x0 = np.array([3, 4])  # Starting from point (3, 4)
path = gradient_descent(df, x0, 0.00005, 300000)
print(f"Minimum point located at (x, y)={tuple(path[-1])}")
# end::beale_function_gradient_descent

path = np.array(path)
fig = plot_graph(path, x, y, z, minima_, xmin, xmax, ymin, ymax)
fig.savefig("assets/beale_function_gradient_descent_path.png")
