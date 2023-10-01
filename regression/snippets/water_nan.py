import warnings

import numpy as np
import scipy.io as sio

from regression.regression import linear_regression_vec

warnings.filterwarnings(category=RuntimeWarning, action="ignore")

dataset = sio.loadmat("regression/snippets/water.mat")
x_train = dataset["X"]
x_val = dataset["Xval"]
x_test = dataset["Xtest"]

# squeeze the target variables into one dimensional arrays
y_train = dataset["y"].squeeze()
y_val = dataset["yval"].squeeze()
y_test = dataset["ytest"].squeeze()

X, y = x_train, y_train

# tag::water_linear_regression_nan
X = np.hstack((X, X**2, X**3))
history = linear_regression_vec(X, y, alpha=0.001, num_iters=5000)
print("w:", history[-1])
# end::water_linear_regression_nan
