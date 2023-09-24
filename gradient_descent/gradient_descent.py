# tag::gradient_descent
def gradient_descent(df, x, alpha=0.01, iterations=100, epsilon=1e-8):
    history = [x]
    for i in range(iterations):
        if abs(df(x)) < epsilon:
            break
        x = x - alpha * df(x)
        history.append(x)
    return history
# end::gradient_descent