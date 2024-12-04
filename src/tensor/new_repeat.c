#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_new_repeat(double repeated_value, index_t shape[], index_t ndim) {
	index_t size   = pascal_tensor_utils_size_from_shape(shape, ndim);

	double* values = malloc(size * sizeof(double));
	for (int i = 0; i < size; i++) {
		values[i] = repeated_value;
	}

	Tensor tensor = pascal_tensor_new_no_malloc(values, shape, ndim);

	return tensor;
}
