import numpy as np

from gradient_descent.gradient_descent import gradient_descent_momentum, plot_graph
from gradient_descent.snippets.gradient_descent_beale_1 import (df, minima_, x, x0, xmax, xmin, y,
                                                                ymax, ymin, z)

# tag::beale_function_gradient_descent
path = gradient_descent_momentum(df, x0, alpha=0.000005, iterations=300000)
print(f"Minimum point located at (x, y)={tuple(path[-1])}")
# end::beale_function_gradient_descent

path = np.asarray(path)
fig = plot_graph(path, x, y, z, minima_, xmin, xmax, ymin, ymax)
fig.savefig("assets/beale_function_gradient_descent_momentum_path.png")
