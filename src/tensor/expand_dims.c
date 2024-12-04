#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_expand_dims(Tensor a, index_t dim) {
	pascal_tensor_assert(dim < a->ndim, "<dim> must be less than <a.ndim>\n");

	index_t  ndim   = a->ndim + 1;
	index_t* shape  = malloc(ndim * sizeof(index_t));
	index_t* stride = malloc(ndim * sizeof(index_t));

	for (int i = 0; i < ndim; i++) {
		if (i < dim) {
			shape[i]  = a->shape[i];
			stride[i] = a->_stride[i];
		} else if (i == dim) {
			shape[i]  = 1;
			stride[i] = a->_stride[i - 1];
		} else {
			shape[i]  = a->shape[i - 1];
			stride[i] = a->_stride[i - 1];
		}
	}

	double* values = malloc(a->size * sizeof(double));
	for (int i = 0; i < a->size; i++) {
		values[i] = a->values[i];
	}

	Tensor tensor   = pascal_tensor_init();
	tensor->ndim    = ndim;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->size    = a->size;
	tensor->values  = values;

	return tensor;
}
