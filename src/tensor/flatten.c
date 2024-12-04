#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_flatten(Tensor a) {
	if (a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	Tensor tensor   = pascal_tensor_init();

	index_t* shape  = malloc(sizeof(index_t));
	shape[0]        = a->size;

	index_t* stride = malloc(sizeof(index_t));
	stride[0]       = 1;

	double* values  = malloc(a->size * sizeof(double));
	for (int i = 0; i < a->size; i++) {
		values[i] = a->values[i];
	}

	tensor->ndim    = 1;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->size    = a->size;
	tensor->values  = values;

	return tensor;
}
