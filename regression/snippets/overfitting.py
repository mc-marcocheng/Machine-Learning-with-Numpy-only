import matplotlib.pyplot as plt
import numpy as np

np.random.seed(896)


def sample(n_samples, std=0.25):
    # sine dataset generation
    x = np.sort(np.random.uniform(0, 1, n_samples))
    y = np.sin(2 * np.pi * x) + np.random.normal(scale=std, size=x.shape)
    return x, y


n_samples = 10
x, y = sample(n_samples)

x_test = np.linspace(0, 1, 100)
xx = x_test
y_test = np.sin(2 * np.pi * x_test)
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.scatter(x, y, facecolor="none", edgecolor="b", s=50, label="training data")
plt.savefig("assets/sine_toy_plot.png")
plt.clf()

# tag::sine_underfitting_overfitting
for i, K in enumerate([0, 1, 3, 9]):
    X = np.array([np.power(x, k) for k in range(K + 1)]).T
    w = np.linalg.inv(X.T @ X) @ (X.T @ y)
    print(f"w for {K}-degree polynomial:", w)
# end::sine_underfitting_overfitting

for i, K in enumerate([0, 1, 3, 9]):
    X = np.array([np.power(x, k) for k in range(K + 1)]).T
    w = np.linalg.inv(X.T @ X) @ (X.T @ y)
    plt.subplot(2, 2, i + 1)
    y_predict = 0
    for i, wi in enumerate(w):
        y_predict += wi * np.power(x_test, i)
    plt.scatter(x, y, facecolor="none", edgecolor="b", s=50, label="training data")
    y_test = np.sin(2 * np.pi * x_test)
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y_predict, c="r", label="fitting")
    plt.ylim(-1.5, 1.5)

plt.savefig("assets/sine_underfitting_overfitting.png")
plt.clf()

# train with more data
n_samples = 100
x, y = sample(n_samples)
K = 9
X = np.array([np.power(x, k) for k in range(K + 1)]).T
w = np.linalg.inv(X.T @ X) @ (X.T @ y)
y_predict = 0
for i, wi in enumerate(w):
    y_predict += wi * np.power(x_test, i)
plt.scatter(x, y, facecolor="none", edgecolor="b", s=50, label="training data")
y_test = np.sin(2 * np.pi * x_test)
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y_predict, c="r", label="fitting")
plt.ylim(-1.5, 1.5)
plt.savefig("assets/sine_more_data.png")
