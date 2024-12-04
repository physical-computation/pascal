import os
import argparse

import numpy as np

x_dim = 1

parser = argparse.ArgumentParser()
parser.add_argument('n_data_points', type=int)
parser.add_argument("n_test_data_points", type=int)

args = parser.parse_args()

N_DATA_POINTS = args.n_data_points
N_TEST_DATA_POINTS = args.n_test_data_points

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def generate_data_nn():
	w0 = np.random.normal(-1.0, 1.0, [x_dim, 3])
	w1 = np.random.uniform(-2.0, 2.0, [3, 1])

	def f(x, w0, w1):
		return np.sin(np.tanh(x@w0)@w1)

	x = np.random.uniform(-np.pi, np.pi, size=(N_DATA_POINTS, 1))
	x_test = np.linspace(-2*np.pi, 2*np.pi, N_TEST_DATA_POINTS).reshape(-1, 1)

	y = f(x, w0, w1) + np.random.normal(0, 0.02, size=(N_DATA_POINTS, 1))
	y_test = f(x_test, w0, w1)

	return x, y, x_test, y_test

def save_data(x, y, location):
	with open(location, "w+") as f:
		np.savetxt(f, np.append(x, y, axis=1).flatten())

def load_data(file_name: str, num_data_points: int, x_dim: int, y_dim: int):
	x_data = []
	y_data = []

	with open(file_name, "r+") as f:
		for i in range(num_data_points):
			for j in range(x_dim):
				x_data.append(float(f.readline()))

			for j in range(y_dim):
				y_data.append(float(f.readline()))

	x = np.array(x_data).reshape(num_data_points, x_dim)
	y = np.array(y_data).reshape(num_data_points, y_dim)

	return x, y

if __name__ == "__main__":
	x, y, x_test, y_test = generate_data_nn()

	save_data(x, y, "data/train_data.dat")
	save_data(x_test, y_test, "data/test_data.dat")
