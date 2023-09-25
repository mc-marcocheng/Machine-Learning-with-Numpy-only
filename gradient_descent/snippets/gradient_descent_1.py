import matplotlib.pyplot as plt
import numpy as np

from gradient_descent.gradient_descent import gradient_descent

f = lambda x: x**3 - 3*x**2 - 9*x + 2
# tag::gradient_descent_1
df = lambda x: 3*x**2 - 6*x - 9
path = gradient_descent(df, 1, iterations=200)
print(f"Minimum point located at x={path[-1]}")
# end::gradient_descent_1

x = np.arange(-3, 4, 0.01)
y = f(x)
plt.plot(x, y)
path_x = np.array(path)
path_y = f(path_x)
plt.quiver(path_x[:-1], path_y[:-1], path_x[1:] - path_x[:-1], path_y[1:] - path_y[:-1], scale_units="xy", angles="xy", scale=1, color="g")
plt.scatter(path[-1], f(path[-1]))
plt.savefig("assets/gradient_descent_plot_1.png")
