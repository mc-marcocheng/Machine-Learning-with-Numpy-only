import matplotlib.pylab as plt
import numpy as np


# tag::sign_activation
def sign(x):
    return np.array(x > 0, dtype=np.int)

def grad_sign(x):
    return np.zeros_like(x)
# end::sign_activation

x = np.arange(-5.0, 5.0, 0.1)
plt.ylim(-0.1, 1.1)
plt.plot(x, sign(x),label="sigmoid")
plt.plot(x, grad_sign(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.savefig("assets/sign_activation.png")

# tag::tanh_activation
def grad_tanh(x):
    a = np.tanh(x)
    return 1 - a**2
# end::tanh_activation

x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, np.tanh(x),label="tanh")
plt.plot(x, grad_tanh(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.savefig("assets/tanh_activation.png")

# tag::relu_activation
def relu(x):
    return np.maximum(0, x)

def grad_relu(x):
    return 1. * (x > 0)
# end::relu_activation

x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, relu(x),label="relu")
plt.plot(x, grad_relu(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.savefig("assets/relu_activation.png")

# tag::leaky_relu_activation
def leakyRelu(x, a=0.2):
    y = np.copy(x)
    y[y < 0] *= a
    return y

def grad_leakyRelu(x, a=0.2):
    return np.clip(x > 0, a, 1.0)
# end::leaky_relu_activation

x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, leakyRelu(x),label="leakrelu")
plt.plot(x, grad_leakyRelu(x),label="derivative")
plt.legend(loc="upper right", frameon=False)
plt.savefig("assets/leaky_relu_activation.png")
