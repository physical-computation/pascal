#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

/**
 * Returns an unravelled tensor from the given tensor. Unravelling changes the values of the tensor to work with the default stride.
 */
Tensor pascal_tensor_utils_unravel(Tensor a) {
	if (a->_transpose_map == NULL || a->_transpose_map_inverse == NULL) {
		return pascal_tensor_copy(a);
	}

	Tensor tensor  = pascal_tensor_init();

	index_t* shape = malloc(sizeof(index_t) * a->ndim);
	for (int i = 0; i < a->ndim; i++) {
		shape[i] = a->shape[i];
	}

	index_t* stride    = pascal_tensor_utils_default_stride(shape, a->ndim);

	double* values     = malloc(a->size * sizeof(double));

	index_t* a_indexes = malloc(a->ndim * sizeof(index_t));
	for (int i = 0; i < a->size; i++) {
		pascal_tensor_utils_index_from_linear_index(a_indexes, i, stride, a->ndim);

		values[i] = pascal_tensor_get(a, a_indexes);
	}
	free(a_indexes);

	tensor->ndim    = a->ndim;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->size    = a->size;
	tensor->values  = values;

	return tensor;
}
