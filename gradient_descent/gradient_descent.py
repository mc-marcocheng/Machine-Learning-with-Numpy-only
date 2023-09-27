import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# tag::plot_graph
def plot_graph(path, x, y, z, minima_, xmin, xmax, ymin, ymax):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
    ax.quiver(
        path[:-1, 0],
        path[:-1, 1],
        path[1:, 0] - path[:-1, 0],
        path[1:, 1] - path[:-1, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color="g",
    )
    ax.plot(*minima_, "r*", markersize=18)
    ax.set_label("$x$")
    ax.set_label("$y$")
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    return fig
# end::plot_graph


# tag::gradient_descent
def gradient_descent(df, x, alpha=0.01, iterations=100, epsilon=1e-8):
    history = [x]
    for _ in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            break
        x = x - alpha * df(x)
        history.append(x)
    return history
# end::gradient_descent


# tag::gradient_descent_momentum
def gradient_descent_momentum(
    df, x, alpha=0.01, gamma=0.8, iterations=100, epsilon=1e-6
):
    history = [x]
    v = np.zeros_like(x)
    for _ in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            break
        v = gamma * v + alpha * df(x)
        x = x - v
        history.append(x)
    return history
# end::gradient_descent_momentum


# tag::gradient_descent_adagrad
def gradient_descent_Adagrad(df, x, alpha=0.01, iterations=100, epsilon=1e-8):
    history = [x]
    gl = np.zeros_like(x)
    for _ in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            break
        grad = df(x)
        gl += grad**2
        x = x - alpha * grad / (np.sqrt(gl) + epsilon)
        history.append(x)
    return history
# end::gradient_descent_adagrad


# tag::gradient_descent_adadelta
def gradient_descent_Adadelta(df, x, alpha=0.1, rho=0.9, iterations=100, epsilon=1e-8):
    history = [x]
    Eg = np.zeros_like(x)
    Edelta = np.zeros_like(x)
    for _ in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            break
        grad = df(x)
        Eg = rho * Eg + (1 - rho) * (grad**2)
        delta = np.sqrt((Edelta + epsilon) / (Eg + epsilon)) * grad
        x = x - alpha * delta
        Edelta = rho * Edelta + (1 - rho) * (delta**2)
        history.append(x)
    return history
# end::gradient_descent_adadelta
