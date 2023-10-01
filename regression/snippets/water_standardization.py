import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from regression.regression import compute_loss_history, linear_regression_vec, plot_history_predict

dataset = sio.loadmat("regression/snippets/water.mat")
x_train = dataset["X"]
x_val = dataset["Xval"]
x_test = dataset["Xtest"]

# squeeze the target variables into one dimensional arrays
y_train = dataset["y"].squeeze()
y_val = dataset["yval"].squeeze()
y_test = dataset["ytest"].squeeze()

X, y = x_train, y_train
X = np.hstack((X, X**2, X**3))

# tag::water_linear_regression_standardization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std
history = linear_regression_vec(X, y, alpha=0.3, num_iters=30000)
print("w:", history[-1])
# end::water_linear_regression_standardization

loss_history = compute_loss_history(X, y, history)
plot_history_predict(X, y, history[-1], loss_history)
plt.savefig("assets/water_standardization.png")
