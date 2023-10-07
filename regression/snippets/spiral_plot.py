import matplotlib.pyplot as plt
import numpy as np

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


X_spiral, y_spiral = gen_spiral_dataset()
plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spiral, s=20, cmap=plt.cm.spring)
plt.savefig("assets/spiral_plot.png")
