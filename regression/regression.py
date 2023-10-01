import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# tag::linear_regression
def linear_regression(x, y, w, b, alpha=0.01, iterations=100, epsilon=1e-9):
    history = []
    for _ in range(iterations):
        dw = np.mean((w * x + b - y) * x)
        db = np.mean(w * x + b - y)
        if abs(dw) < epsilon and abs(db) < epsilon:
            break
        w -= alpha * dw
        b -= alpha * db
        history.append([w, b])
    return history
# end::linear_regression


# tag::linear_regression_vec
def linear_regression_vec(X, y, alpha, num_iters, gamma=0.8, epsilon=1e-8):
    history = []
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    num_features = X.shape[1]
    v = np.zeros(num_features)
    w = np.zeros(num_features)
    for _ in range(num_iters):
        predictions = X @ w
        errors = predictions - y
        gradient = X.T @ errors / len(y)
        if np.max(np.abs(gradient)) < epsilon:
            break
        v = gamma * v + alpha * gradient
        w = w - v
        history.append(w)
    return history
# end::linear_regression_vec


def loss(x, y, w, b):
    return np.mean((x * w + b - y) ** 2) / 2


def draw_line(plt, w, b, x, linewidth=2):
    m = len(x)
    f = [0] * m
    for i in range(m):
        f[i] = b + w * x[i]
    plt.plot(x, f, linewidth)


def plot_history(x, y, history, figsize=(20, 10)):
    w = [e[0] for e in history]
    b = [e[1] for e in history]
    xmin, xmax, xstep = min(w) - 0.2, max(w) + 0.2, 0.2
    ymin, ymax, ystep = min(b) - 0.2, max(b) + 0.2, 0.2
    ws, bs = np.meshgrid(
        np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep)
    )
    zs = np.array([loss(x, y, w, b) for w, b in zip(np.ravel(ws), np.ravel(bs))])
    z = zs.reshape(ws.shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("$w[0]$", labelpad=30, fontsize=24, fontweight="bold")
    ax.set_ylabel("$w[1]$", labelpad=30, fontsize=24, fontweight="bold")
    ax.set_zlabel("$L(w,b)$", labelpad=30, fontsize=24, fontweight="bold")
    ax.plot_surface(ws, bs, z, rstride=1, cstride=1, color="b", alpha=0.2)

    w_start, b_start, w_end, b_end = (
        history[0][0],
        history[0][1],
        history[-1][0],
        history[-1][1],
    )
    ax.plot(
        [w_start],
        [b_start],
        [loss(x, y, w_start, b_start)],
        markerfacecolor="b",
        markeredgecolor="b",
        marker="o",
        markersize=7,
    )
    ax.plot(
        [w_end],
        [b_end],
        [loss(x, y, w_end, b_end)],
        markerfacecolor="r",
        markeredgecolor="r",
        marker="o",
        markersize=7,
    )

    z2 = [loss(x, y, w, b) for w, b in history]
    ax.plot(
        w, b, z2, markerfacecolor="r", markeredgecolor="r", marker=".", markersize=2
    )
    fig.suptitle("L(w,b)", fontsize=24, fontweight="bold")
    return ws, bs, z


def compute_loss_history(X, y, w_history):
    loss_history = []
    for w in w_history:
        errors = X @ w[1:] + w[0] - y
        loss_history.append((errors**2).mean() / 2)
    return loss_history
