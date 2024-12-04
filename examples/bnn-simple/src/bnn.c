#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "pascal_autodiff.h"
#include "pascal.h"

Tensor bnn_pascal_tensor_squeeze(Tensor a) {
	index_t  ndim  = a->ndim - 1;
	index_t* shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		shape[i] = a->shape[i + 1];
	}

	Tensor t = pascal_tensor_new(a->values, shape, ndim);

	free(shape);
	return t;
}

AutodiffNode bnn_log_gaussian(AutodiffNode w, AutodiffNode mean, AutodiffNode std) {
	AutodiffNode constant_part         = pascal_autodiff_constant_scalar(0.5 * log(2 * M_PI)); // 0.5 * log(2 * pi)

	AutodiffNode half                  = pascal_autodiff_constant_scalar(0.5);
	AutodiffNode negative_one          = pascal_autodiff_constant_scalar(-1.0);

	AutodiffNode diff                  = pascal_autodiff_subtract(w, mean);       // w - mean
	AutodiffNode std_inv               = pascal_autodiff_reciprocal(std);         // 1 / std
	AutodiffNode mean_part_root        = pascal_autodiff_multiply(diff, std_inv); // (w - mean) / std
	AutodiffNode mean_part_square      = pascal_autodiff_multiply(half, pascal_autodiff_square(mean_part_root));

	AutodiffNode log_std               = pascal_autodiff_log(std); // log(std)

	AutodiffNode log_gaussian_unscaled = pascal_autodiff_add(log_std, pascal_autodiff_add(mean_part_square, constant_part)); // log(std) + 0.5 * ((w - mean) / std)^2 + 0.5 * log(2 * pi)

	AutodiffNode log_gaussian          = pascal_autodiff_multiply(negative_one, log_gaussian_unscaled);

	AutodiffNode log_gaussian_clamped  = pascal_autodiff_clamp(log_gaussian, -23.025850929940457, 0);

	return log_gaussian_clamped;
}

AutodiffNode bnn_log_gaussian_mean(AutodiffNode w, AutodiffNode mean) {
	AutodiffNode negative_half    = pascal_autodiff_constant_scalar(-0.5);

	AutodiffNode diff             = pascal_autodiff_subtract(w, mean); // w - mean
	AutodiffNode mean_part_square = pascal_autodiff_square(diff);
	AutodiffNode mean_part        = pascal_autodiff_multiply(negative_half, mean_part_square);

	return mean_part;
}

static Tensor load_weights(char filename[], index_t shape[], index_t ndim) {
	FILE* fp           = fopen(filename, "r+");

	index_t size       = pascal_tensor_utils_size_from_shape(shape, ndim);
	double* values     = malloc(sizeof(double) * size);

	char*  line_buffer = NULL;
	size_t _line_size;
	for (int i = 0; i < size; i++) {
		if (getline(&line_buffer, &_line_size, fp) != -1) {
			sscanf(line_buffer, "%lf", &values[i]);
		} else {
			printf("Error: Could not read line from file.");
		}
	}

	free(line_buffer);
	fclose(fp);

	Tensor tensor = pascal_tensor_new(values, shape, ndim);
	free(values);

	return tensor;
}

AutodiffNode* bnn_init_weights(char type[], index_t sizes[], index_t input_dim, index_t output_dim, index_t n_layers) {
	AutodiffNode* weights = malloc(sizeof(AutodiffNode) * (n_layers + 1));

	char* filename        = malloc(sizeof(char*) * 100);

	// first layer
	index_t n_rows        = input_dim;
	index_t n_cols        = sizes[0];
	sprintf(filename, "data/w_%s_%d.dat", type, 0);
	weights[0] = pascal_autodiff_parameter(load_weights(filename, (index_t[]){n_rows, n_cols}, 2));

	// middle layers
	for (int i = 0; i < n_layers - 1; i++) {
		n_rows = sizes[i];
		n_cols = sizes[i + 1];

		sprintf(filename, "data/w_%s_%d.dat", type, i + 1);
		weights[i + 1] = pascal_autodiff_parameter(load_weights(filename, (index_t[]){n_rows, n_cols}, 2));
	}

	// last layer
	n_rows = sizes[n_layers - 1];
	n_cols = output_dim;
	sprintf(filename, "data/w_%s_%u.dat", type, n_layers);

	weights[n_layers] = pascal_autodiff_parameter(load_weights(filename, (index_t[]){n_rows, n_cols}, 2));

	free(filename);
	return weights;
}

AutodiffNode* bnn_init_epsilons(AutodiffNode w_means[], index_t n_layers) {
	AutodiffNode* epsilons = malloc(sizeof(AutodiffNode) * (n_layers + 1));

	for (int i = 0; i < n_layers + 1; i++) {
		epsilons[i] = pascal_autodiff_constant_scalar(0.0);
	}

	return epsilons;
}

AutodiffNode* bnn_calculate_weights(AutodiffNode w_means[], AutodiffNode w_stds[], AutodiffNode epsilons[], index_t n_layers) {
	AutodiffNode* weights = malloc(sizeof(AutodiffNode) * (n_layers + 1));

	for (int i = 0; i < n_layers + 1; i++) {
		AutodiffNode w = pascal_autodiff_add(w_means[i], pascal_autodiff_multiply(w_stds[i], epsilons[i]));
		weights[i]     = w;
	}

	return weights;
}

AutodiffNode* bnn_calculate_stds(AutodiffNode w_rhos[], AutodiffNode epsilons[], index_t n_layers) {
	AutodiffNode one   = pascal_autodiff_constant_scalar(1.0);

	AutodiffNode* stds = malloc(sizeof(AutodiffNode) * (n_layers + 1));

	for (int i = 0; i < n_layers + 1; i++) {
		AutodiffNode w_std = pascal_autodiff_log(pascal_autodiff_add(one, pascal_autodiff_exp(w_rhos[i])));
		stds[i]            = w_std;
	}

	return stds;
}

AutodiffNode bnn_mse(AutodiffNode y, AutodiffNode y_out, AutodiffNode w, AutodiffNode w_mean, AutodiffNode w_std) {
	AutodiffNode prior_mean      = pascal_autodiff_constant_scalar(0.0);
	AutodiffNode prior_std       = pascal_autodiff_constant_scalar(1.0);

	AutodiffNode log_prior       = pascal_autodiff_sum_all(bnn_log_gaussian(w, prior_mean, prior_std));

	AutodiffNode std_prob        = bnn_log_gaussian(w, w_mean, w_std);
	AutodiffNode log_variational = pascal_autodiff_sum_all(std_prob);

	AutodiffNode log_likelihood  = pascal_autodiff_sum_all(bnn_log_gaussian_mean(y, y_out));

	AutodiffNode loss            = pascal_autodiff_subtract(log_variational, pascal_autodiff_add(log_likelihood, log_prior));

	return loss;
}

void bnn_sample_epsilons(AutodiffNode epsilons[], index_t n_layers) {
	for (int i = 0; i < n_layers + 1; i++) {
		pascal_tensor_free(epsilons[i]->forward);

		epsilons[i]->forward = pascal_tensor_random_uniform(0.0, 1.0, (index_t[]){1}, 1);
	}
}

void bnn_update_weights(AutodiffNode weights, double learning_rate) {
	Tensor weighted_gradient = pascal_tensor_scalar_multiply(weights->grad, learning_rate);
	Tensor new_weights       = pascal_tensor_subtract(weights->forward, weighted_gradient);

	pascal_tensor_free(weights->forward);
	pascal_tensor_free(weighted_gradient);

	weights->forward = new_weights;
}

void bnn_save_tensor(FILE* location, Tensor tensor) {
	for (int i = 0; i < tensor->size; i++) {
		fprintf(location, "%.*e\n", DECIMAL_DIG, tensor->values[i]);
	}
}

void bnn_save_loss_values(char* location, double loss_values[], index_t n_values) {
	FILE* file = fopen(location, "wb");
	for (int i = 0; i < n_values; i++) {
		fprintf(file, "%.*e\n", DECIMAL_DIG, loss_values[i]);
	}
	fclose(file);
}
