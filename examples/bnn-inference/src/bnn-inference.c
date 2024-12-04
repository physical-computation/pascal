#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "pascal.h"

void load_means_and_stddevs(char location[], double** weight_means, double** weight_stddevs, double** bias_means, double** bias_stddevs, index_t nlayers, index_t weights_sizes[], index_t biases_sizes[]) {
	for (int i = 0; i < nlayers; i++) {
		char weight_mu_path[100];
		char weight_sigma_path[100];
		char bias_mu_path[100];
		char bias_sigma_path[100];

		sprintf(weight_mu_path, "%s/layers.%d.weight_mu.csv", location, i);
		sprintf(weight_sigma_path, "%s/layers.%d.weight_sigma.csv", location, i);
		sprintf(bias_mu_path, "%s/layers.%d.bias_mu.csv", location, i);
		sprintf(bias_sigma_path, "%s/layers.%d.bias_sigma.csv", location, i);

		FILE* weight_mu_file    = fopen(weight_mu_path, "r+");
		FILE* weight_sigma_file = fopen(weight_sigma_path, "r+");
		FILE* bias_mu_file      = fopen(bias_mu_path, "r+");
		FILE* bias_sigma_file   = fopen(bias_sigma_path, "r+");

		index_t weight_size     = weights_sizes[i];
		index_t bias_size       = biases_sizes[i];

		double* weights_mu      = malloc(sizeof(double) * weight_size);
		double* weights_sigma   = malloc(sizeof(double) * weight_size);
		double* biases_mu       = malloc(sizeof(double) * bias_size);
		double* biases_sigma    = malloc(sizeof(double) * bias_size);

		for (int j = 0; j < weight_size; j++) {
			fscanf(weight_mu_file, "%lf", &weights_mu[j]);
			fscanf(weight_sigma_file, "%lf", &weights_sigma[j]);
		}

		for (int j = 0; j < bias_size; j++) {
			fscanf(bias_mu_file, "%lf", &biases_mu[j]);
			fscanf(bias_sigma_file, "%lf", &biases_sigma[j]);
		}

		weight_means[i]   = weights_mu;
		weight_stddevs[i] = weights_sigma;

		bias_means[i]     = biases_mu;
		bias_stddevs[i]   = biases_sigma;
	}
}

double relu(double x) {
	return x > 0 ? x : 0;
}

Tensor forward(Tensor x, Tensor weights[], Tensor biases[], index_t nlayers) {
	for (int i = 0; i < nlayers - 1; i++) {
		Tensor w = pascal_tensor_transpose(weights[i], (index_t[]){1, 0});
		x        = pascal_tensor_matmul(x, w);
		x        = pascal_tensor_add(x, biases[i]);
		x        = pascal_tensor_map(x, relu);

		pascal_tensor_free(w);
	}

	Tensor w = pascal_tensor_transpose(weights[nlayers - 1], (index_t[]){1, 0});
	x        = pascal_tensor_matmul(x, w);
	x        = pascal_tensor_add(x, biases[nlayers - 1]);

	pascal_tensor_free(w);
	return x;
}

void save_samples(char location[], double samples[], int N) {
	char filename[100];

	sprintf(filename, "%s/pascal.csv", location);
	FILE* file = fopen(filename, "w+");

	for (int i = 0; i < N; i++) {
		fprintf(file, "%lf\n", samples[i]);
	}

	fclose(file);
}

int main(int argc, char* argv[]) {
	struct timeval _sto, _sta;
	gettimeofday(&_sta, NULL);
	srand(time(NULL) + _sta.tv_usec);

	int N;

	index_t nlayers          = 4;

	double** weight_means    = malloc(sizeof(double*) * nlayers);
	double** weight_stddevs  = malloc(sizeof(double*) * nlayers);
	double** bias_means      = malloc(sizeof(double*) * nlayers);
	double** bias_stddevs    = malloc(sizeof(double*) * nlayers);

	index_t weights_sizes[4] = {16, 16 * 16, 16 * 16, 16};
	index_t biases_sizes[4]  = {16, 16, 16, 1};

	load_means_and_stddevs("model_params", weight_means, weight_stddevs, bias_means, bias_stddevs, nlayers, weights_sizes, biases_sizes);

	Tensor* weights      = malloc(sizeof(Tensor) * nlayers);
	Tensor* biases       = malloc(sizeof(Tensor) * nlayers);

	index_t shapes[4][2] = {
			{16, 1},
			{16, 16},
			{16, 16},
			{1, 16}};

	N               = atoi(argv[1]);
	Tensor x        = pascal_tensor_new((double[]){-0.2}, (index_t[]){1, 1}, 2);

	double* samples = malloc(sizeof(double) * N);

	for (int iteration = 0; iteration < N; iteration++) {
		for (int i = 0; i < nlayers; i++) {
			weights[i] = pascal_tensor_uncertain_normal(weight_means[i], weight_stddevs[i], shapes[i], 2);
			biases[i]  = pascal_tensor_uncertain_normal(bias_means[i], bias_stddevs[i], shapes[i], 1);
		}
		Tensor y           = forward(x, weights, biases, nlayers);
		samples[iteration] = y->values[0];

		pascal_tensor_free(y);
	}

	save_samples("samples", samples, N);
}
