#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_append(Tensor a, Tensor b, index_t axis) {
	pascal_tensor_assert(a->ndim == b->ndim, "Orders must be equal\n");

	index_t* shape = malloc(a->ndim * sizeof(index_t));
	for (int i = 0; i < a->ndim; i++) {
		if (i != axis) {
			pascal_tensor_assert(a->shape[i] == b->shape[i], "Shapes must be equal expect on the specified axis\n");
			shape[i] = a->shape[i];
		} else {
			shape[i] = a->shape[i] + b->shape[i];
		}
	}

	index_t  size    = pascal_tensor_utils_size_from_shape(shape, a->ndim);
	index_t* stride  = pascal_tensor_utils_default_stride(shape, a->ndim);

	double*  values  = malloc(size * sizeof(double));
	index_t* indexes = malloc(a->ndim * sizeof(index_t));
	for (int i = 0; i < size; i++) {
		pascal_tensor_utils_index_from_linear_index(indexes, i, stride, a->ndim);
		if (indexes[axis] < a->shape[axis]) {
			values[i] = pascal_tensor_get(a, indexes);
		} else {
			indexes[axis] = indexes[axis] - a->shape[axis];
			values[i]     = pascal_tensor_get(b, indexes);
		}
	}
	free(indexes);

	Tensor tensor   = pascal_tensor_init();
	tensor->size    = size;
	tensor->ndim    = a->ndim;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->values  = values;

	return tensor;
}
