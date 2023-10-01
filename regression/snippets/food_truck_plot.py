import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("regression/snippets/food_truck_data.txt", delimiter=",")
train_x = data[:, 0]
train_y = data[:, 1]

fig, ax = plt.subplots()
ax.scatter(train_x, train_y, marker="x", c="red")
plt.title("Food Truck Dataset", fontsize=16)
plt.xlabel("City Population in 10000s", fontsize=14)
plt.ylabel("Food Truck Profit in 10000s", fontsize=14)
plt.axis([4, 25, -5, 25])
plt.savefig("assets/food_truck_plot.png")
