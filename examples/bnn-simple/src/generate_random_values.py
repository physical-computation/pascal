import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('learning_rate', type=float)
parser.add_argument('n', type=int)
parser.add_argument("n_test_data_points", type=int)
parser.add_argument("n_repetitions", type=int)
parser.add_argument("n_data_points", type=int)
parser.add_argument("n_nodes", type=int)
parser.add_argument("n_layers", type=int)

args = parser.parse_args()

N_NODES = args.n_nodes
N_LAYERS = args.n_layers
N = args.n
N_TEST_DATA_POINTS = args.n_test_data_points
N_DATA_POINTS = args.n_data_points
N_REPETITIONS = args.n_repetitions
NUM_BATCHES = 1
CLASSES = 1
PI = 0.25
SAMPLES = 1
x_dim = 1
y_dim = 1

n_epsilons_single = (N + (N_REPETITIONS) * 2) * 2 * (N_LAYERS + 1)


if __name__ == '__main__':
	# generate epsilons
	with open("data/epsilons.dat", "w+") as f:
		np.savetxt(f, np.random.normal(0, 1, n_epsilons_single).flatten())

	for i in range(N_LAYERS + 1):
		# input layer
		if i == 0:
			w_mean = np.random.normal(0, 1, (x_dim, N_NODES))
			w_rho = np.random.normal(0, 1, (x_dim, N_NODES))

			np.savetxt("data/w_mean_{}.dat".format(i), w_mean.flatten())
			np.savetxt("data/w_rho_{}.dat".format(i), w_rho.flatten())

		# output layer
		elif i == N_LAYERS:
			w_mean = np.random.normal(0, 1, (N_NODES, y_dim))
			w_rho = np.random.normal(0, 1, (N_NODES, y_dim))

			np.savetxt("data/w_mean_{}.dat".format(i), w_mean.flatten())
			np.savetxt("data/w_rho_{}.dat".format(i), w_rho.flatten())

		# hidden layers
		else:
			w_mean = np.random.normal(0, 1, (N_NODES, N_NODES))
			w_rho = np.random.normal(0, 1, (N_NODES, N_NODES))

			np.savetxt("data/w_mean_{}.dat".format(i), w_mean.flatten())
			np.savetxt("data/w_rho_{}.dat".format(i), w_rho.flatten())
