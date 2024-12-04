#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_random_normal(double mean, double variance, index_t shape[], index_t ndim) {
	Tensor   tensor = pascal_tensor_init();
	index_t  size   = pascal_tensor_utils_size_from_shape(shape, ndim);
	index_t* stride = pascal_tensor_utils_default_stride(shape, ndim);

	index_t* _shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		pascal_tensor_assert(shape[i] != 0, "Shape cannot contain 0\n");
		_shape[i] = shape[i];
	}

	double* values = malloc(size * sizeof(double));
	for (int i = 0; i < size; i++) {
		values[i] = pascal_tensor_random_sample_normal(mean, variance);
	}

	tensor->shape   = _shape;
	tensor->ndim    = ndim;
	tensor->size    = size;
	tensor->_stride = stride;
	tensor->values  = values;

	return tensor;
}
