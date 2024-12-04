#include <assert.h>
#include <bnn.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "pascal_autodiff.h"
#include "pascal.h"

index_t EPSILON_INDEX = 0;
index_t EPSILON_COUNTER;

void update(AutodiffNode w_means[], AutodiffNode w_rhos[], index_t N_LAYERS, double learning_rate) {
	for (int i = 0; i < N_LAYERS + 1; i++) {
		bnn_update_weights(w_means[i], learning_rate);
		bnn_update_weights(w_rhos[i], learning_rate);
	}
}

double* load_epsilons(char* location, index_t N_LAYERS, index_t n, index_t n_test_data_points, index_t n_repetitions) {
	index_t n_epsilons = (n + ((n_repetitions) * 2)) * (N_LAYERS + 1);
	FILE*   fp         = fopen(location, "r+");
	double* epsilons   = malloc(n_epsilons * sizeof(double));

	char*  line_buffer = NULL;
	size_t _line_size;
	for (int i = 0; i < n_epsilons; i++) {
		if (getline(&line_buffer, &_line_size, fp) != -1) {
			sscanf(line_buffer, "%lf", &epsilons[i]);
		} else {
			printf("Error: Could not read line from file.");
		}
	}

	free(line_buffer);
	fclose(fp);

	for (int i = 0; i < n_epsilons; i++) {
		if (i > 100) {
			continue;
		}
	}
	return epsilons;
}

void sample_epsilons(AutodiffNode epsilons[], index_t N_LAYERS, double* EPSILONS) {
	for (int i = 0; i < N_LAYERS + 1; i++) {
		pascal_tensor_free(epsilons[i]->forward);
		epsilons[i]->forward = pascal_tensor_new((double[]){EPSILONS[EPSILON_INDEX]}, (index_t[]){1}, 1);

		EPSILON_INDEX++;
		EPSILON_COUNTER++;
	}
}

AutodiffNode forward(AutodiffNode x, AutodiffNode weights[], AutodiffNodeOperation activations[], index_t N_LAYERS) {
	// first layer
	x = pascal_autodiff_matmul(x, weights[0]);
	x = pascal_autodiff_operate(activations[0], x);

	// middle layers
	for (int i = 1; i < N_LAYERS + 1; i++) {
		x = pascal_autodiff_matmul(x, weights[i]);
		x = pascal_autodiff_operate(activations[i], x);
	}

	return x;
}

AutodiffNode loss_function(AutodiffNode y, AutodiffNode y_out, AutodiffNode weights[], AutodiffNode w_means[], AutodiffNode w_stds[], index_t N_LAYERS) {
	AutodiffNode prior_mean      = pascal_autodiff_constant_scalar(0.0);
	AutodiffNode prior_std       = pascal_autodiff_constant_scalar(1.0);

	AutodiffNode log_prior       = pascal_autodiff_constant_scalar(0.0);
	AutodiffNode log_variational = pascal_autodiff_constant_scalar(0.0);

	for (int i = 0; i < N_LAYERS + 1; i++) {
		AutodiffNode lg       = bnn_log_gaussian(weights[i], prior_mean, prior_std);
		log_prior             = pascal_autodiff_add(log_prior, pascal_autodiff_sum_all(lg));

		AutodiffNode std_prob = bnn_log_gaussian(weights[i], w_means[i], w_stds[i]);
		log_variational       = pascal_autodiff_add(log_variational, pascal_autodiff_sum_all(std_prob));
	}

	AutodiffNode log_likelihood = pascal_autodiff_sum_all(bnn_log_gaussian_mean(y, y_out));

	AutodiffNode loss           = pascal_autodiff_subtract(log_variational, pascal_autodiff_add(log_likelihood, log_prior));

	return loss;
}

void evaluate(char* load_location, char* save_location, AutodiffNodeOperation activations[], AutodiffNode w_means[], AutodiffNode w_rhos[], AutodiffNode epsilons[], index_t N_LAYERS, index_t x_dim, index_t y_dim, index_t n_test_data_points, index_t n_repetitions, double* EPSILONS) {
	Tensor x_test = NULL;
	Tensor y_test = NULL;

	load_pascal_tensor_data(load_location, &x_test, &y_test, n_test_data_points, x_dim, y_dim);

	AutodiffNode  x       = pascal_autodiff_new(x_test);
	AutodiffNode* w_stds  = bnn_calculate_stds(w_rhos, epsilons, N_LAYERS);
	AutodiffNode* weights = bnn_calculate_weights(w_means, w_stds, epsilons, N_LAYERS);
	AutodiffNode  y_out   = forward(x, weights, activations, N_LAYERS);

	FILE* file            = fopen(save_location, "wb");
	for (int i = 0; i < n_repetitions; i++) {
		sample_epsilons(epsilons, N_LAYERS, EPSILONS);

		pascal_autodiff_compute_forward(y_out);
		bnn_save_tensor(file, y_out->forward);
	}
	fclose(file);
}

int main(int argc, char* argv[]) {
	srand(time(NULL));

	if (argc != 9) {
		printf("Incorrect number of arguments");
		return 1;
	}

	char* eptr;

	double  learning_rate;
	index_t N;
	index_t N_TEST_DATA_POINTS;
	index_t N_REPETITIONS;
	index_t N_DATA_POINTS;
	index_t N_NODES;
	index_t N_LAYERS;
	index_t PRINT_FREQUENCY;

	learning_rate                      = strtod(argv[1], &eptr);
	N                                  = atoi(argv[2]);
	N_TEST_DATA_POINTS                 = atoi(argv[3]);
	N_REPETITIONS                      = atoi(argv[4]);
	N_DATA_POINTS                      = atoi(argv[5]);
	N_NODES                            = atoi(argv[6]);
	N_LAYERS                           = atoi(argv[7]);
	PRINT_FREQUENCY                    = atoi(argv[8]);

	index_t x_dim                      = 1;
	index_t y_dim                      = 1;

	AutodiffNodeOperation* activations = malloc((N_LAYERS + 1) * sizeof(AutodiffNodeOperation));
	activations[N_LAYERS]              = AutodiffNodeOperationSin;
	for (int i = 0; i < N_LAYERS; i++) {
		activations[i] = AutodiffNodeOperationTanh;
	}

	index_t* layers = malloc((N_LAYERS) * sizeof(index_t));
	for (int i = 0; i < N_LAYERS; i++) {
		layers[i] = N_NODES;
	}

	Tensor x_data = NULL;
	Tensor y_data = NULL;
	load_pascal_tensor_data("data/train_data.dat", &x_data, &y_data, N_DATA_POINTS, x_dim, y_dim);

	double* EPSILONS           = load_epsilons("data/epsilons.dat", N_LAYERS, N, N_TEST_DATA_POINTS, N_REPETITIONS);

	AutodiffNode x             = pascal_autodiff_new(x_data);
	AutodiffNode y             = pascal_autodiff_new(y_data);

	// initialize weight means and rhos
	AutodiffNode* w_means      = bnn_init_weights("mean", layers, x_dim, y_dim, N_LAYERS);
	AutodiffNode* w_rhos       = bnn_init_weights("rho", layers, x_dim, y_dim, N_LAYERS);
	AutodiffNode* epsilons     = bnn_init_epsilons(w_means, N_LAYERS);

	// calculate weights
	AutodiffNode* w_stds       = bnn_calculate_stds(w_rhos, epsilons, N_LAYERS);

	AutodiffNode* weights      = bnn_calculate_weights(w_means, w_stds, epsilons, N_LAYERS);

	// network
	AutodiffNode y_out         = forward(x, weights, activations, N_LAYERS);

	AutodiffNode loss          = loss_function(y, y_out, weights, w_means, w_stds, N_LAYERS);

	double*        loss_values = (double*)malloc(N * sizeof(double));
	struct timeval stop, start;
	for (int i = 0; i < N; i++) {
		gettimeofday(&start, NULL);
		sample_epsilons(epsilons, N_LAYERS, EPSILONS);

		pascal_autodiff_compute_forward(loss);
		pascal_autodiff_compute_backward(loss);

		update(w_means, w_rhos, N_LAYERS, learning_rate);

		gettimeofday(&stop, NULL);
		long int completion_time = ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec));

		loss_values[i]           = pascal_tensor_get(loss->forward, (index_t[]){0});
		if (i % PRINT_FREQUENCY == 0) {
			printf("iteration=%-5d\t\tloss=%15.8lf\t\ttime=%lu\xC2\xB5s\n", i + 1, pascal_tensor_get(loss->forward, (index_t[]){0}), completion_time);
		}
	}

	evaluate("data/test_data.dat", "results/bnn-eval.dat", activations, w_means, w_rhos, epsilons, N_LAYERS, x_dim, y_dim, N_TEST_DATA_POINTS, N_REPETITIONS, EPSILONS);

	pascal_autodiff_free(loss);

	free(w_means);
	free(w_rhos);
	free(epsilons);
	free(w_stds);
	free(loss_values);
	free(activations);
	free(weights);
	free(EPSILONS);
	free(layers);
}
