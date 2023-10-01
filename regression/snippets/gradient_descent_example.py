import matplotlib.pyplot as plt
import numpy as np

from regression.regression import draw_line, linear_regression, plot_history

data = np.loadtxt("regression/snippets/food_truck_data.txt", delimiter=",")
train_x = data[:, 0]
train_y = data[:, 1]

X = train_x
y = train_y

# tag::food_truck_gradient_descent
w, b = 0, 0
history = linear_regression(X, y, w, b, alpha=0.02, iterations=1000)
w, b = history[-1]
print(f"{w=}, {b=}")
# end::food_truck_gradient_descent

plt.scatter(X, y, marker="x", c="red")
plt.title("Food Truck Dataset", fontsize=16)
plt.xlabel("City Population in 10000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10000s", fontsize=14)
plt.axis([4, 25, -5, 25])
draw_line(plt, w, b, X, 6)
plt.savefig("assets/food_truck_linear_regression_plot.png")

plot_history(X, y, history)
plt.savefig("assets/food_truck_linear_regression_loss.png")
