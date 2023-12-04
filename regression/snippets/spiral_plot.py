import matplotlib.pyplot as plt
import numpy as np

from regression.regression import gradient_descent_softmax

np.random.seed(100)


def gen_spiral_dataset(N=100, D=2, K=3):
    """Generates spiral toy dataset.
    N is the number of points per class;
    P is the dimensionality;
    K is the number of classes
    """
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype="uint8")
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


# tag::spiral_gen
X_spiral, y_spiral = gen_spiral_dataset()
# end::spiral_gen
plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spiral, s=20, cmap=plt.cm.spring)
plt.savefig("assets/spiral_plot.png")
plt.clf()

# tag::spiral_gradient_descent
w = np.zeros([X_spiral.shape[1] + 1, len(np.unique(y_spiral))])
w_history = gradient_descent_softmax(
    w, X_spiral, y_spiral, reg=1e-3, alpha=1.0, iterations=200
)
# end::spiral_gradient_descent

# plot the resulting classifier
h = 0.02
x_min, x_max = X_spiral[:, 0].min() - 1, X_spiral[:, 0].max() + 1
y_min, y_max = X_spiral[:, 1].min() - 1, X_spiral[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = np.dot(np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()], w)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spiral, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.savefig("assets/spiral_classify.png")
