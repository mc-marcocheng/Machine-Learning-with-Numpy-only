# Gradient Descent
Gradient descent (also often called steepest descent) is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.

## Basics
Given a function $f(x)$,
$$f(x+\Delta x)-f(x)\approx f^\prime(x)\Delta x$$

Gradient descent algorithm:<br>
Starting from an initial `x`, we update the value of `x` using the following formula:
$$x=x-\alpha f^\prime(x)$$

Code for gradient descent:
```python
def gradient_descent(df, x, alpha=0.01, iterations=100, epsilon=1e-8):
    history = [x]
    for _ in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            break
        x = x - alpha * df(x)
        history.append(x)
    return history
```

For example, we want to find the local minimum of $f(x)=x^3-3x^2-9x+2$. Its derivative is $f^\prime(x)=3x^2-6x-9$.
```python
df = lambda x: 3 * x**2 - 6 * x - 9
path = gradient_descent(df, 1, iterations=200)
print(f"Minimum point located at x={path[-1]}")
```
<details open>
<summary>Output</summary>

```
Minimum point located at x=2.999999999256501
```

</details>


![](../assets/gradient_descent_plot_1.png)

Similarly, for multivariate functions:
$$f(x+\Delta x)-f(x)\approx\nabla f(x)\Delta x$$

Example: [Beale's function](https://www.sfu.ca/~ssurjano/beale.html)
$$f(x,y)=(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2$$

![](../assets/beale_function.png)

Its derivatives:

$$\frac{\partial f(x,y)}{\partial x}=2(1.5-x+xy)(y-1)+2(2.25-x+xy^2)(y^2-1)+2(2.625-x+xy^3)(y^3-1)$$

$$\frac{\partial f(x,y)}{\partial y}=2(1.5-x+xy)x+2(2.25-x+xy^2)(2yx)+2(2.625-x+xy^3)(3y^2x)$$

```python
df = lambda x: np.array(
    [
        2 * (1.5 - x[0] + x[0] * x[1]) * (x[1] - 1)
        + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (x[1] ** 2 - 1)
        + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (x[1] ** 3 - 1),
        2 * (1.5 - x[0] + x[0] * x[1]) * x[0]
        + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (2 * x[0] * x[1])
        + 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (3 * x[0] * x[1] ** 2),
    ]
)
x0 = np.array([3, 4])  # Starting from point (3, 4)
path = gradient_descent(df, x0, 0.00005, 300000)
print(f"Minimum point located at (x, y)={tuple(path[-1])}")
```
<details open>
<summary>Output</summary>

```
Minimum point located at (x, y)=(2.996394053883693, 0.4990985442431246)
```

</details>


Note: Beale's function is a gradient descent testing function, meaning that it is not easy for gradient descent algorithms. The learning rate $\alpha$ and iterations need to be tuned manually.

Its gradient descent path:
![](../assets/beale_function_gradient_descent_path.png)

## Momentum
In momentum-based gradient descent, we will consider the previous updates in derivatives.

Let $v_{t-1}$ be the last update vector. The current update vector will be:
$$v_t=\gamma v_{t-1}+\alpha\nabla f(x)$$

The update vector is used to update the current `x`:
$$x=x-v_t$$

$v_t$ is called the **momentum**.

```python
code
```
