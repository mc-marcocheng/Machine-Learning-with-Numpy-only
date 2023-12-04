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


def plot_history_predict(X, y, w, loss_history, fig_size=(12, 4)):
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("Loss")
    plt.xlabel("iterations")

    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    x = X[:, 1]

    predicts = X @ w
    plt.subplot(1, 2, 2)
    plt.scatter(x, predicts)

    indices = x.argsort()
    sorted_x = x[indices[::-1]]
    sorted_predicts = predicts[indices[::-1]]
    plt.plot(sorted_x, sorted_predicts, color="red")
    plt.scatter(x, y)
    plt.title("Prediction")


# tag::gradient_descent_reg
def gradient_descent_reg(X, y, reg, alpha, num_iters, gamma=0.8, epsilon=1e-8):
    w_history = []
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    num_features = X.shape[1]
    v = np.zeros_like(num_features)
    w = np.zeros(num_features)
    for _ in range(num_iters):
        gradient = X.T @ (X @ w - y) / len(y) + 2 * reg * w
        if np.max(np.abs(gradient)) < epsilon:
            break
        v = gamma * v + alpha * gradient
        w = w - v
        w_history.append(w)
    return w_history
# end::gradient_descent_reg


def loss_reg(w, X, y, reg=0):
    errors = X @ w[1:] + w[0] - y
    reg_error = reg * np.sum(np.square(w))
    return (errors**2).mean() / 2 + reg_error


def compute_loss_history_reg(X, y, w_history, reg=0):
    loss_history = []
    for w in w_history:
        loss_history.append(loss_reg(w, X, y, reg))
    return loss_history


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# tag::gradient_descent_logistic_regression
def gradient_descent_logistic_reg(
    X, y, lambda_, alpha, num_iters, gamma=0.8, epsilon=1e-8
):
    w_history = []
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    num_features = X.shape[1]
    v = np.zeros_like(num_features)
    w = np.zeros(num_features)
    for _ in range(num_iters):
        gradient = (sigmoid(X @ w) - y).T @ X / len(y)
        gradient += 2 * lambda_ * w
        if np.max(np.abs(gradient)) < epsilon:
            break
        v = gamma * v + alpha * gradient
        w = w - v
        w_history.append(w)
    return w_history
# end::gradient_descent_logistic_regression


def loss_logistic(w, X, y, reg=0.0):
    f = sigmoid(X @ w[1:] + w[0])
    loss = -np.mean((np.log(f).T * y + np.log(1 - f).T * (1 - y)))
    loss += reg * (np.sum(np.square(w)))
    return loss


def loss_history_logistic(w_history, X, y, reg=0.0):
    loss_history = []
    for w in w_history:
        loss_history.append(loss_logistic(w, X, y, reg))
    return loss_history


# tag::softmax
def softmax(x):
    a = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - a)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
# end::softmax


# tag::softmax_gradient
def softmax_gradient(z):
    f = softmax(z)
    grad = -np.outer(f, f) + np.diag(f.flatten())
    return grad
# end::softmax_gradient


# tag::softmax_backward
def softmax_backward(z, df):
    grad = softmax_gradient(z)
    return df @ grad
# end::softmax_backward


# tag::cross_entropy_loss
def cross_entropy(F, y):
    m = len(F)  # number of samples
    log_Fy = -np.log(F[range(m), y])
    return np.sum(log_Fy) / m
# end::cross_entropy_loss


# tag::cross_entropy_one_hot
def cross_entropy_one_hot(F, Y):
    m = len(F)
    return -np.sum(Y * np.log(F)) / m
# end::cross_entropy_one_hot


# tag::softmax_cross_entropy
def softmax_cross_entropy(Z, y):
    m = len(Z)
    F = softmax(Z)
    log_Fy = -np.log(F[range(m), y])
    return np.sum(log_Fy) / m
# end::softmax_cross_entropy


# tag::softmax_cross_entropy_one_hot
def softmax_cross_entropy_one_hot(Z, y):
    F = softmax(Z)
    loss = -np.sum(y * np.log(F), axis=1)
    return np.mean(loss)
# end::softmax_cross_entropy_one_hot


# tag::softmax_cross_entropy_gradient
def grad_softmax_cross_entropy(Z, y):
    F = softmax(Z)
    F[range(len(Z)), y] -= 1
    return F / len(Z)
# end::softmax_cross_entropy_gradient


# tag::softmax_cross_entropy_gradient_one_hot
def grad_softmax_cross_entropy_one_hot(Z, y):
    F = softmax(Z)
    return (F - y) / len(Z)
# end::softmax_cross_entropy_gradient_one_hot


# tag::gradient_softmax
def gradient_softmax(W, X, y, reg):
    Z = X @ W
    I_i = np.zeros_like(Z)
    I_i[np.arange(len(Z)), y] = 1
    F = softmax(Z)
    grad = np.dot(X.T, F - I_i) / len(X) + 2 * reg * W
    return grad
# end::gradient_softmax


# tag::loss_softmax
def loss_softmax(W, X, y, reg):
    Z = X @ W
    Z_i_y_i = Z[np.arange(len(Z)), y]
    negative_log_prob = -Z_i_y_i + np.log(np.sum(np.exp(Z), axis=-1))
    loss = np.mean(negative_log_prob) + reg * np.sum(W * W)
    return loss
# end::loss_softmax


# tag::gradient_softmax_onehot
def gradient_softmax_onehot(W, X, y, reg):
    Z = X @ W
    F = softmax(Z)
    grad = np.dot(X.T, F - y) / len(X) + 2 * reg * W
    return grad
# end::gradient_softmax_onehot


# tag::loss_softmax_onehot
def loss_softmax_onehot(W, X, y, reg):
    Z = X @ W
    F = softmax(Z)
    loss = -np.sum(y * np.log(F)) / len(X) + reg * np.sum(W * W)
    return loss
# end::loss_softmax_onehot


# tag::softmax_gradient_descent
def gradient_descent_softmax(
    w, X, y, reg=0.0, alpha=0.01, iterations=100, epsilon=1e-8
):
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    w_history = []
    for i in range(iterations):
        gradient = gradient_softmax(w, X, y, reg)
        if np.max(np.abs(gradient)) < epsilon:
            print("gradient is small enough!")
            print("iterated num is :", i)
            break
        w = w - (alpha * gradient)
        w_history.append(w)
    return w_history
# end::softmax_gradient_descent


def getAccuracy(w, X, y):
    X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))
    probs = softmax(np.dot(X, w))
    predicts = np.argmax(probs, axis=1)
    accuracy = sum(predicts == y) / (float(len(y)))
    return accuracy
