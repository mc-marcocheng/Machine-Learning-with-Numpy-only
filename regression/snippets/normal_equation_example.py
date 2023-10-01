import numpy as np

data = np.loadtxt("regression/snippets/food_truck_data.txt", delimiter=",")
train_x = data[:, 0]
train_y = data[:, 1]

# tag::food_truck_normal_equation
X = np.ones((len(train_x), 2))
X[:, 1] = train_x
y = train_y

b, w = np.linalg.inv(X.T @ X) @ (X.T @ y)
print(f"{w=}, {b=}")
# end::food_truck_normal_equation
