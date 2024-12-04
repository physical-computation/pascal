import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot')
parser.add_argument("n_test_data_points", type=int)
parser.add_argument("n_repetitions", type=int)
parser.add_argument("n_data_points", type=int)

args = parser.parse_args()

N_TEST_DATA_POINTS = args.n_test_data_points
N_REPETITIONS = args.n_repetitions
N_DATA_POINTS = args.n_data_points

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


def load_results(file_name: str, n_test_data_pointss: int, n_repetitions: int):
	with open(file_name, "r+") as f:
		for i in range(n_repetitions):
			y_i = []

			for j in range(n_test_data_pointss):
				y_i.append(float(f.readline()))

			y_i = np.array(y_i).reshape(1,  n_test_data_pointss)

			y = np.append(y, y_i, axis=0) if i != 0 else y_i
	return y


def plot_data(location: str, name: str, x: np.array, y: np.array, x_test: np.array, y_test: np.array, pred_mean: np.array, pred_std: np.array):
	plt.figure()
	plt.fill_between(x_test, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std,
						color='gray', alpha=.5, label='+/- 3 std')
	plt.scatter(x, y, marker='x', c='black', label='target')
	plt.plot(x_test, pred_mean, c='red', label='Prediction')
	plt.plot(x_test, y_test, c='grey', label='truth')

	plt.legend()
	plt.title(name)
	plt.tight_layout()

	plt.savefig("plots/" + location, format='pdf')


if __name__ == '__main__':
	x, y = load_data("data/train_data.dat", N_DATA_POINTS, 1, 1)
	x = x.reshape((1, N_DATA_POINTS))
	y = y.reshape((1, N_DATA_POINTS))

	x_test, y_test = load_data("data/test_data.dat", N_TEST_DATA_POINTS, 1, 1)
	x_test = x_test.squeeze(1)
	y_test = y_test.squeeze(1)

	y_pred = load_results("results/bnn-eval.dat", N_TEST_DATA_POINTS, N_REPETITIONS)
	pred_std = y_pred.std(axis=0)
	pred_mean = y_pred.mean(axis=0)
	plot_data("bnn-eval.pdf", "Pascal", x, y, x_test, y_test, pred_mean, pred_std)

	# y_pred = load_results("results/bnn-eval-initial.dat", N_TEST_DATA_POINTS, N_REPETITIONS)
	# pred_std = y_pred.std(axis=0)
	# pred_mean = y_pred.mean(axis=0)
	# plot_data("bnn-eval-initial.pdf", x, y, x_test, y_test, pred_mean, pred_std)


	y_pred = load_results("results/bnn-eval-pytorch.dat", N_TEST_DATA_POINTS, N_REPETITIONS)
	pred_std = y_pred.std(axis=0)
	pred_mean = y_pred.mean(axis=0)
	plot_data("bnn-eval-pytorch.pdf", "PyTorch", x, y, x_test, y_test, pred_mean, pred_std)





