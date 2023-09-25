import numpy as np

from gradient_descent.gradient_descent import gradient_descent_momentum, plot_graph

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

# tag::beale_function_gradient_descent
path = gradient_descent_momentum(df, x0, alpha=0.000005, iterations=300000)
print(f"Minimum point located at (x, y)={tuple(path[-1])}")
# end::beale_function_gradient_descent

path = np.asarray(path)
fig = plot_graph(path, x, y, z, minima_, xmin, xmax, ymin, ymax)
fig.savefig("assets/beale_function_gradient_descent_momentum_path.png")
