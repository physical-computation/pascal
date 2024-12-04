import os

import numpy as np

input_d = 3
n = 100

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def generate_data(n: int = n) -> None:
	x = np.random.uniform(-10, 10, [n, 3]).reshape((n, 3))

	w1 = np.random.uniform(-10, 10, [3, 10])
	w2 = np.random.uniform(-10, 10, [10, 1])

	x1 = sigmoid(x@w1)
	y = x1@w2

	y += np.random.normal(0, 0.1, (n, 1))

	try:
		os.remove("data/data.dat")
	except FileNotFoundError:
		pass
	with open("data/data.dat", "a") as f:
		np.savetxt(f, np.append(x, y, axis=1).flatten())

if __name__ == "__main__":
	generate_data()