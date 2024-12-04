#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_random_uniform(double min, double max, index_t shape[], index_t ndim) {
	Tensor   tensor = pascal_tensor_init();
	index_t  size   = pascal_tensor_utils_size_from_shape(shape, ndim);
	index_t* stride = pascal_tensor_utils_default_stride(shape, ndim);

	index_t* _shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		pascal_tensor_assert(shape[i] != 0, "Shape cannot contain 0; what would that even mean?\n");
		_shape[i] = shape[i];
	}

	double* values = malloc(size * sizeof(double));
	for (int i = 0; i < size; i++) {
		values[i] = pascal_tensor_random_sample_uniform(min, max);
	}

	tensor->shape   = _shape;
	tensor->ndim    = ndim;
	tensor->size    = size;
	tensor->_stride = stride;
	tensor->values  = values;

	return tensor;
}
