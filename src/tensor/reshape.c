#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_reshape(Tensor a, index_t new_shape[], index_t ndim) {
	pascal_tensor_assert(a->size == pascal_tensor_utils_size_from_shape(new_shape, ndim), "New shape isn't compatible with the given tensor\n");

	if (a->_transpose_map != NULL || a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	index_t size    = a->size;

	index_t* stride = pascal_tensor_utils_default_stride(new_shape, ndim);
	index_t* shape  = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		pascal_tensor_assert(new_shape[i] != 0, "New shape cannot contain 0\n");
		shape[i] = new_shape[i];
	}

	Tensor tensor  = pascal_tensor_init();

	double* values = malloc(size * sizeof(double));
	for (int i = 0; i < size; i++) {
		values[i] = a->values[i];
	}

	tensor->size    = size;
	tensor->ndim    = ndim;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->values  = values;
	tensor->shape   = shape;

	return tensor;
}
