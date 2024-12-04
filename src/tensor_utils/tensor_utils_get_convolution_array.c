#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

double* pascal_tensor_utils_get_convolution_array(Tensor a, index_t filter_shape[], index_t filter_size, index_t filter_ndim, index_t start_index[]) {
	double* values          = malloc(sizeof(double) * filter_size);

	index_t* _start_indexes = malloc(a->ndim * sizeof(index_t));
	for (int i = 0; i < a->ndim; i++) {
		_start_indexes[i] = start_index[i];
	}

	index_t* _stride = pascal_tensor_utils_default_stride(filter_shape, filter_ndim);

	index_t* indexes = malloc(filter_ndim * sizeof(index_t));
	for (int i = 0; i < filter_size; i++) {
		pascal_tensor_utils_index_from_linear_index(indexes, i, _stride, filter_ndim);

		for (int i = 0; i < filter_ndim; i++) {
			_start_indexes[a->ndim - i - 1] = start_index[a->ndim - i - 1] + indexes[filter_ndim - i - 1];
		}

		values[i] = pascal_tensor_get(a, _start_indexes);
	}
	free(indexes);
	free(_stride);
	free(_start_indexes);

	return values;
}
