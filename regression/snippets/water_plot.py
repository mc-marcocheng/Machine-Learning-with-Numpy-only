import matplotlib.pyplot as plt
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

plt.scatter(x_train, y_train, marker="x", s=40, c="red")
plt.scatter(x_val, y_val, marker="o", s=40, c="blue")
plt.xlabel("change in water level", fontsize=14)
plt.ylabel("water flowing out of the dam", fontsize=14)
plt.title("Training sample", fontsize=16)
plt.savefig("assets/water_dataset.png")
plt.clf()

# tag::water_linear_regression
X, y = x_train, y_train
history = linear_regression_vec(X, y, alpha=0.001, num_iters=5000)
print("w:", history[-1])
# end::water_linear_regression

w = history[-1]
loss_history = compute_loss_history(X, y, history)
plot_history_predict(X, y, w, loss_history)
plt.savefig("assets/water_linear_regression.png")
plt.clf()
