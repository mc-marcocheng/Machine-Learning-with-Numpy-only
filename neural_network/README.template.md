# Neural Network Introduction

## Perceptron
Perceptron is a simple function that can perform binary classification. Logistic regression uses the sigmoid function $\sigma$ to produce a probability $\sigma(xw)$. Perceptron on the other hand uses a threshold value to produce a value either 0 or 1: $\text{sgn}_b(xw)$.

$$\text{sgn}_b(z)=\left\lbrace\begin{aligned}1 && \text{if }z\geq b \\
0 && \text{otherwise}\end{aligned}\right.$$

Denote $f_w(x)=\text{sgn}_b(xw)$ as the perceptron function, where $x$ is the input and $w$ is the weight. Then,

$$f_w(x)=\left\lbrace\begin{aligned}1 && \text{if }\sum_jw_jx_j\geq b \\
0 && \text{otherwise}\end{aligned}\right.$$

which can be rewritten into:

$$f_w(x)=\left\lbrace\begin{aligned}1 && \text{if }\sum_jw_jx_j+b\geq 0 \\
0 && \text{otherwise}\end{aligned}\right.=\text{sgn}(xw+b)$$

## Neuron
Neuron accepts multiple inputs, performs weighted sum and then through a linear or non-linear activation function, producing one or more output values.

$$a=g\left(\sum_jw_jx_j\right)$$

$x_j$ is an input, $w_j$ is its respective weight, and $g$ is the activation function.

## Activation Functions

$\text{sign}(x)$:

```python
${{ sign_activation }}
```

$[[ -neural_network.snippets.activation_functions ]]

![](../assets/sign_activation.png)

$\tanh(x)$:

$$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

$$\tanh^\prime(x)=1-\tanh^2(x)$$

```python
${{ tanh_activation }}
```

![](../assets/tanh_activation.png)

ReLU (Rectified Linear Unit):

$$\text{ReLU}(x)=\left\lbrace\begin{aligned}x && \text{if } x>0 \\
0 && \text{otherwise}\end{aligned}\right.$$

$$\text{ReLU}'(x)=\left\lbrace\begin{aligned}1 && \text{if } x>0 \\
0 && \text{otherwise}\end{aligned}\right.$$

```python
${{ relu_activation }}
```

![](../assets/relu_activation.png)

Leaky ReLU:

$$\text{LeakyReLU}(x)=\left\lbrace\begin{aligned}x && \text{if } x>0 \\
ax && \text{otherwise}\end{aligned}\right.$$

where a is a small constant (typically 0.01).

$$\text{LeakyReLU}'(x)=\left\lbrace\begin{aligned}1 && \text{if } x>0 \\
a && \text{otherwise}\end{aligned}\right.$$

```python
${{ leaky_relu_activation }}
```

![](../assets/leaky_relu_activation.png)

## Neural Network

A neural network is composed of many perceptrons to represent a complicated function. It has an input layer and an output layer. In between, there are one or more hidden layers.

$[[ -neural_network.snippets.neural_network_plot ]]

![](../assets/neural_network_networkx.png)

Take the above 2-layered neural network as an example. It defines a function $f: \mathbb{R}^D \rightarrow \mathbb{R}^K$, where D is the dimensionality of input and K is the dimensionality of vector $f(x)$. Assume that the activation functions for the first layer are all $g^{[1]}$. Each perception in the hidden layer takes $x=(x_1, x_2)$ and produces an output. Their outputs form a vector $a^{[1]}=(a_1^{[1]}\quad a_2^{[1]}\quad a_3^{[1]}\quad a_4^{[1]})$.

$$a_1^{[1]} = g^{[1]}(x_1W_{11}^{[1]} + x_2W_{21}^{[1]}+b_1^{[1]})$$

$$a_2^{[1]} = g^{[1]}(x_1W_{12}^{[1]} + x_2W_{22}^{[1]}+b_2^{[1]})$$

$$a_3^{[1]} = g^{[1]}(x_1W_{13}^{[1]} + x_2W_{23}^{[1]}+b_3^{[1]})$$

$$a_4^{[1]} = g^{[1]}(x_1W_{14}^{[1]} + x_2W_{24}^{[1]}+b_4^{[1]})$$

Write

$$W^{[1]}=\left[\begin{matrix}W_{11}^{[1]} & W_{12}^{[1]} & W_{13}^{[1]} & W_{14}^{[1]} \\
W_{21}^{[1]} & W_{22}^{[1]} & W_{23}^{[1]} & W_{24}^{[1]}\end{matrix}\right]$$

$$b^{[1]}=(b_1^{[1]}\quad b_2^{[1]}\quad b_3^{[1]}\quad b_4^{[1]})$$

Then the output of the first layer can be written as $a^{[1]}=g^{[1]}(xW^{[1]}+b^{[1]})$.

Similarly for the second layer, we write

$$W^{[2]}=\left[\begin{matrix}W_{11}^{[2]} & W_{12}^{[2]} & W_{13}^{[2]} \\
W_{21}^{[2]} & W_{22}^{[2]} & W_{23}^{[2]} \\
W_{31}^{[2]} & W_{32}^{[2]} & W_{33}^{[2]} \\
W_{41}^{[2]} & W_{42}^{[2]} & W_{43}^{[2]} \end{matrix}\right]$$

$$b^{[2]}=(b_1^{[2]}\quad b_2^{[2]}\quad b_3^{[2]})$$

The output of the second layer can be written as $a^{[2]}=g^{[2]}(a^{[1]}W^{[2]}+b^{[2]})$.

The whole neural network can be represented as $f(x)$:

$$f(x)=g^{[2]}\left(\left(g^{[1]}(xW^{[1]}+b^{[1]})\right)W^{[2]}+b^{[2]}\right)$$

The calculation sequence $x\rightarrow z^{[1]}\rightarrow a^{[1]}\rightarrow z^{[2]}\rightarrow a^{[2]}$ is called forward propagation.

For $m$ samples $x^{(i)}$, their features form a matrix $X$:

$$X=\begin{bmatrix}x^{(1)}\\
x^{(2)}\\
\vdots\\
x^{(m)}\end{bmatrix}$$

Let

$$Z^{[l]}=\begin{bmatrix}z^{(1)[l]} \\
z^{(2)[l]} \\
\vdots \\
z^{(m)[l]}\end{bmatrix},\;A^{[l]}=\begin{bmatrix}a^{(1)[l]} \\
a^{(2)[l]} \\
\vdots \\
a^{(m)[l]}\end{bmatrix}$$

where $z^{(i)[l]}$ is the weighted sum for sample $i$ in $l$-th layer, and $a^{(i)[l]}$ is the activation value for sample $i$ in $l$-th layer.

The forward propagation of the samples can then be written as

$$\begin{aligned}Z^{[l]}&=A^{[l-1]}W^{[l]}+b^{[l]}\\
A^{[l]}&=g^{[l]}(Z^{[l]})\end{aligned}$$

Example:

```python
${{ forward_propagation }}
```

$[[ neural_network.snippets.forward_propagation ]]

#
