#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "pascal_autodiff.h"
#include "pascal.h"

Tensor pascal_tensor_squeeze(Tensor a) {
	index_t  ndim  = a->ndim - 1;
	index_t* shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		shape[i] = a->shape[i + 1];
	}

	Tensor t = pascal_tensor_new(a->values, shape, ndim);

	free(shape);
	return t;
}

void update_weights(AutodiffNode weights, Tensor gradient, double learning_rate) {
	Tensor s_gradient        = pascal_tensor_squeeze(gradient);

	Tensor weighted_gradient = pascal_tensor_scalar_multiply(s_gradient, learning_rate);
	Tensor new_weights       = pascal_tensor_subtract(weights->forward, weighted_gradient);

	*(weights->forward)      = *new_weights;
}

// Single layer neural network
AutodiffNode forward(AutodiffNode x, AutodiffNode w0, AutodiffNode w1) {
	AutodiffNode x1    = pascal_autodiff_matmul(x, w0);
	AutodiffNode s1    = pascal_autodiff_sigmoid(x1);

	AutodiffNode y_out = pascal_autodiff_matmul(s1, w1);

	return y_out;
}

// Mean squared error loss function
AutodiffNode mse(AutodiffNode y_out, AutodiffNode y) {
	AutodiffNode loss_diff   = pascal_autodiff_subtract(y_out, y);
	AutodiffNode loss_square = pascal_autodiff_square(loss_diff);
	AutodiffNode loss        = pascal_autodiff_mean_all(loss_square);

	return loss;
}

int main() {
	// Setup
	srand(time(NULL));
	index_t N_ITERATIONS    = 100;
	double  learning_rate   = 0.05;

	Tensor x_data           = NULL;
	Tensor y_data           = NULL;

	index_t num_data_points = 50;
	index_t x_dim           = 3;
	index_t y_dim           = 1;

	index_t n_nodes         = 10;

	// Load data
	load_pascal_tensor_data("data/data.dat", &x_data, &y_data, num_data_points, x_dim, y_dim);

	AutodiffNode x     = pascal_autodiff_new(x_data);
	AutodiffNode y     = pascal_autodiff_new(y_data);

	// Initialize weights
	AutodiffNode w0    = pascal_autodiff_random_uniform_parameter(-1.0, 1.0, (index_t[]){x_dim, n_nodes}, 2);
	AutodiffNode w1    = pascal_autodiff_random_uniform_parameter(-1.0, 1.0, (index_t[]){n_nodes, y_dim}, 2);

	// NN
	AutodiffNode y_out = forward(x, w0, w1);

	// Loss
	AutodiffNode loss  = mse(y_out, y);

	for (int i = 0; i < N_ITERATIONS; i++) {
		pascal_autodiff_compute_forward(loss);
		pascal_autodiff_compute_backward(loss);

		update_weights(w0, w0->grad, learning_rate);
		update_weights(w1, w1->grad, learning_rate);

		printf("iteration = %-3d\t\tloss = %lf\n", i, pascal_tensor_get(loss->forward, (index_t[]){0}));
	}
}
