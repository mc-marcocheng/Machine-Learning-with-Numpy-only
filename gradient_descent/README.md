# Gradient Descent
## Basics
Starting from an initial `x`, we update the value of `x` using the following formula:
$$x=x-\alpha f^\prime(x)$$

Code for gradient descent:
```python
def gradient_descent(df, x, alpha=0.01, iterations=100, epsilon=1e-8):
    history = [x]
    for _ in range(iterations):
        if abs(df(x)) < epsilon:
            break
        x = x - alpha * df(x)
        history.append(x)
    return history
```
