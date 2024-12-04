#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "assert.h"
#include "pascal.h"

double noisy_sin(double x) {
	return sin(x) + 0.5 * (rand() / (double)RAND_MAX);
}

double squared_exponential(double x1, double x2, double l, double sigma_f) {
	double diff = x1 - x2;
	return sigma_f * sigma_f * exp(-1 * diff * diff / (l * l));
}

// only works for the 1D case
Tensor kernel(Tensor data, double l, double sigma_f) {
	index_t* _shape  = malloc(sizeof(index_t) * 2);
	double*  _values = malloc(sizeof(double) * data->shape[0] * data->shape[0]);

	_shape[0]        = data->shape[0];
	_shape[1]        = _shape[0];

	for (int i = 0; i < data->shape[0]; i++) {
		for (int j = 0; j < data->shape[0]; j++) {
			if (i <= j) {
				double v                        = squared_exponential(data->values[i], data->values[j], l, sigma_f);
				_values[i * data->shape[0] + j] = v;
				_values[j * data->shape[0] + i] = v;
			}
		}
	}

	Tensor k = pascal_tensor_new_no_malloc(_values, _shape, 2);
	;
	return k;
}

Tensor kernel_general(Tensor x1, Tensor x2, double l, double sigma_f) {
	index_t* _shape  = malloc(sizeof(index_t) * 2);
	double*  _values = malloc(sizeof(double) * x1->shape[0] * x2->shape[0]);

	_shape[0]        = x1->shape[0];
	_shape[1]        = x2->shape[0];

	for (int i = 0; i < x1->size; i++) {
		for (int j = 0; j < x2->size; j++) {
			double v                  = squared_exponential(x1->values[i], x2->values[j], l, sigma_f);
			_values[i * x2->size + j] = v;
		}
	}

	Tensor k = pascal_tensor_new_no_malloc(_values, _shape, 2);
	return k;
}

Tensor mean_pred(Tensor x_new, Tensor x, Tensor y, Tensor data_kern, double l, double sigma_f, double noise) {
	Tensor k_new = kernel_general(x_new, x, l, sigma_f);

	// Tensor t_noise   = pascal_tensor_eye(k->shape[0]);
	// Tensor k_noise   = pascal_tensor_add(k, pascal_tensor_scalar_multiply(t_noise, noise));
	// Tensor data_kern = pascal_tensor_linalg_solve(k_noise, y);

	Tensor rv    = pascal_tensor_matmul(k_new, data_kern);

	pascal_tensor_free(k_new);

	return rv;
}

Tensor variance_pred(Tensor x_new, Tensor x, Tensor y, Tensor k_noise, double l, double sigma_f, double noise) {
	Tensor k_new_left    = kernel_general(x_new, x, l, sigma_f);
	Tensor k_new_right   = pascal_tensor_transpose(k_new_left, (index_t[]){1, 0});

	Tensor k_new         = kernel(x_new, l, sigma_f);

	// Tensor t_noise       = pascal_tensor_eye(k->shape[0]);
	// Tensor k_noise       = pascal_tensor_add(k, pascal_tensor_scalar_multiply(t_noise, noise));
	Tensor data_kern     = pascal_tensor_linalg_solve(k_noise, k_new_right);

	Tensor right_summand = pascal_tensor_matmul(k_new_left, data_kern);
	Tensor full_mat      = pascal_tensor_subtract(k_new, right_summand);
	Tensor rv            = pascal_tensor_diag(full_mat);

	pascal_tensor_free(k_new_left);
	pascal_tensor_free(k_new_right);
	pascal_tensor_free(data_kern);
	pascal_tensor_free(right_summand);
	pascal_tensor_free(k_new);

	return rv;
}

void save_data(Tensor x, Tensor y, Tensor x_new, Tensor means, Tensor variances) {
	FILE* fp = fopen("results/results.json", "w");
	fprintf(fp, "{\n");

	fprintf(fp, "\t\"n\": %u,\n", x->shape[0]);
	fprintf(fp, "\t\"n_new\": %u,\n", x_new->shape[0]);
	fprintf(fp, "\t\"x\": [\n");
	for (int i = 0; i < x->size; i++) {
		if (i == x->size - 1) {
			fprintf(fp, "\t\t%lf\n", x->values[i]);
		} else {
			fprintf(fp, "\t\t%lf,\n", x->values[i]);
		}
	}
	fprintf(fp, "\t],\n");

	fprintf(fp, "\t\"y\": [\n");
	for (int i = 0; i < y->size; i++) {
		if (i == y->size - 1) {
			fprintf(fp, "\t\t%lf\n", y->values[i]);
		} else {
			fprintf(fp, "\t\t%lf,\n", y->values[i]);
		}
	}
	fprintf(fp, "\t],\n");

	fprintf(fp, "\t\"x_new\": [\n");
	for (int i = 0; i < x_new->size; i++) {
		if (i == x_new->size - 1) {
			fprintf(fp, "\t\t%lf\n", x_new->values[i]);
		} else {
			fprintf(fp, "\t\t%lf,\n", x_new->values[i]);
		}
	}
	fprintf(fp, "\t],\n");

	fprintf(fp, "\t\"means\": [\n");
	for (int i = 0; i < means->size; i++) {
		if (i == means->size - 1) {
			fprintf(fp, "\t\t%lf\n", means->values[i]);
		} else {
			fprintf(fp, "\t\t%lf,\n", means->values[i]);
		}
	}
	fprintf(fp, "\t],\n");
	fprintf(fp, "\t\"variances\": [\n");
	for (int i = 0; i < variances->size; i++) {
		if (i == variances->size - 1) {
			fprintf(fp, "\t\t%lf\n", variances->values[i]);
		} else {
			fprintf(fp, "\t\t%lf,\n", variances->values[i]);
		}
	}
	fprintf(fp, "\t]\n");
	fprintf(fp, "}\n");
	fclose(fp);
}

int main() {
	srand(time(NULL));

	index_t n        = 10;
	index_t n_new    = 1000;
	double  l        = 1.7;
	double  sigma_f  = 1.0;
	double  noise    = 0.01;

	Tensor x         = pascal_tensor_new((double[]){-9.42477796, -7.33038286, -5.23598776, -3.14159265, -1.04719755, 1.04719755, 3.14159265, 5.23598776, 7.33038286, 9.42477796},
	                                     (index_t[]){10, 1}, 2);
	Tensor y         = pascal_tensor_new((double[]){0.15707481, -2.0195747, 1.59070617, 0.48162433, -0.9591348, 0.81104839, 0.49939094, -1.1432049, 1.82739085, 0.17157255},
	                                     (index_t[]){10, 1}, 2);

	Tensor k         = kernel(x, l, sigma_f);
	Tensor t_noise   = pascal_tensor_eye(k->shape[0]);
	Tensor k_noise   = pascal_tensor_add(k, pascal_tensor_scalar_multiply(t_noise, pow(noise, 2)));
	Tensor data_kern = pascal_tensor_linalg_solve(k_noise, y);

	Tensor x_new     = pascal_tensor_linspace(-3 * M_PI, 3 * M_PI, n_new);

	Tensor means     = mean_pred(x_new, x, y, data_kern, l, sigma_f, noise);
	Tensor variances = variance_pred(x_new, x, y, k_noise, l, sigma_f, noise);

	save_data(x, y, x_new, means, variances);

	pascal_tensor_free(x);
	pascal_tensor_free(y);
	pascal_tensor_free(k);
	pascal_tensor_free(x_new);
	pascal_tensor_free(means);
	pascal_tensor_free(variances);

	return 0;
}
