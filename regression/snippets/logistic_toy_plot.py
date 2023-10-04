import matplotlib.pyplot as plt
import numpy as np

from regression.regression import gradient_descent_logistic_reg, loss_history_logistic

np.random.seed(0)

n_pts = 100
D = 2
Xa = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts)])
Xb = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts)])

X = np.append(Xa, Xb, axis=1).T
y = np.append(np.zeros(n_pts), np.ones(n_pts)).T

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(X[:n_pts, 0], X[:n_pts, 1], color="lightcoral", label="$Y = 0$")
ax.scatter(X[n_pts:, 0], X[n_pts:, 1], color="blue", label="$Y = 1$")
ax.set_title("Sample Dataset")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(loc=4)
fig.savefig("assets/logistic_toy_plot.png")
fig.clf()

# tag::logistic_regression_toy
w_history = gradient_descent_logistic_reg(
    X, y, lambda_=0.0, alpha=0.01, num_iters=10000
)
w = w_history[-1]
print("w:", w)
# end::logistic_regression_toy

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# tag::logistic_regression_decision_boundary
x1 = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
x2 = -w[0] / w[2] - x1 * w[1] / w[2]
ax[0].plot(x1, x2, color="k", ls="--", lw=2)
# end::logistic_regression_decision_boundary

ax[0].scatter(
    X[: int(n_pts), 0], X[: int(n_pts), 1], color="lightcoral", label="$y = 0$"
)
ax[0].scatter(X[int(n_pts) :, 0], X[int(n_pts) :, 1], color="blue", label="$y = 1$")
ax[0].set_title("$x_1$ vs. $x_2$")
ax[0].set_xlabel("$x_1$")
ax[0].set_ylabel("$x_2$")
ax[0].legend(loc=4)

loss_history = loss_history_logistic(w_history, X, y, reg=0.0)
ax[1].plot(loss_history, color="r")
ax[1].set_ylim(0, ax[1].get_ylim()[1])
ax[1].set_title("$L(w)$ vs. Iteration")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("$L(w)$")

fig.tight_layout()
fig.savefig("assets/logistic_toy_decision_boundary.png")
