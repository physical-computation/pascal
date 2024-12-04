#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

int load_data(char file_name[], double** x, double** y) {
	FILE* fp = fopen(file_name, "r+");

	long  num_data_points;
	char  _num_lines[10];
	char* _;

	// get the number of data points from the first line of file.
	fgets(_num_lines, 10, fp);
	num_data_points   = strtol(_num_lines, &_, 10);

	// resize x and y to accommodate dataset.
	*x                = realloc(*x, num_data_points * sizeof(double));
	*y                = realloc(*y, num_data_points * sizeof(double));

	// scan each line, and allocate to x and y buffers.
	int   i           = 0;
	char* line_buffer = (char*)calloc(2048, sizeof(char));
	if (line_buffer == NULL) {
		printf("Dynamic memory allocation failed for line_buffer.\n");
		exit(-1);
	}

	while ((fgets(line_buffer, 2048, fp) != NULL) &&
	       (i < num_data_points)) {
		sscanf(line_buffer, "%lf %lf", &(*x)[i], &(*y)[i]);
		i++;
	}

	free(line_buffer);
	fclose(fp);
	return i;
}

void load_pascal_tensor_data(char file_name[], Tensor* x, Tensor* y, index_t num_data_points, index_t x_dim, index_t y_dim) {
	FILE* fp           = fopen(file_name, "r+");

	// resize x and y to accommodate dataset.
	// *x = realloc(*x, sizeof(Tensor));
	// *y = realloc(*y, sizeof(Tensor));

	double* values_x   = malloc(num_data_points * x_dim * sizeof(double));
	double* values_y   = malloc(num_data_points * y_dim * sizeof(double));

	index_t* shape_x   = malloc(2 * sizeof(index_t));
	index_t* shape_y   = malloc(2 * sizeof(index_t));

	shape_x[0]         = num_data_points;
	shape_x[1]         = x_dim;

	shape_y[0]         = num_data_points;
	shape_y[1]         = y_dim;

	// scan each line, and allocate to x and y buffers.
	char*  line_buffer = NULL;
	size_t _line_size;

	for (int i = 0; i < num_data_points; i++) {
		int j = 0;
		for (int j = 0; j < x_dim; j++) {
			if (getline(&line_buffer, &_line_size, fp) != -1) {
				sscanf(line_buffer, "%lf", &values_x[(i * x_dim) + j]);
			} else {
				printf("Error: Could not read line from file.");
			}
		}

		j = 0;
		for (int j = 0; j < y_dim; j++) {
			if (getline(&line_buffer, &_line_size, fp) != -1) {
				sscanf(line_buffer, "%lf", &values_y[(i * y_dim) + j]);

				// printf("%lu, %lu, %lu\n", i, j, i*y_dim + j);
			} else {
				printf("Error: Could not read line from file.");
			}
		}
	}

	*x = pascal_tensor_new_no_malloc(values_x, shape_x, 2);
	*y = pascal_tensor_new_no_malloc(values_y, shape_y, 2);

	free(line_buffer);
	free(shape_x);
	free(shape_y);
	fclose(fp);
}
