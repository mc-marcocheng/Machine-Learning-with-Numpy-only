import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from regression.regression import (compute_loss_history_reg, gradient_descent_reg,
                                   plot_history_predict)

dataset = sio.loadmat("regression/snippets/water.mat")
x_train = dataset["X"]

# squeeze the target variables into one dimensional arrays
y_train = dataset["y"].squeeze()

# tag::water_regularization
x_train_n = np.hstack(tuple(x_train ** (i + 1) for i in range(9)))
train_means = x_train_n.mean(axis=0)
train_stdevs = np.std(x_train_n, axis=0, ddof=1)
x_train_n = (x_train_n - train_means) / train_stdevs

history = gradient_descent_reg(x_train_n, y_train, reg=0.2, alpha=0.3, num_iters=100000)
print("w:", history[-1])
# end::water_regularization

loss_history = compute_loss_history_reg(x_train_n, y_train, history, reg=0.2)
plot_history_predict(x_train_n, y_train, history[-1], loss_history)
plt.savefig("assets/water_regularization.png")
