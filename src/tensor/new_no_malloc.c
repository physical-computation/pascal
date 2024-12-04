#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_new_no_malloc(double values[], index_t shape[], index_t ndim) {
	Tensor tensor   = pascal_tensor_init();

	index_t size    = pascal_tensor_utils_size_from_shape(shape, ndim);

	index_t* _shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		_shape[i] = shape[i];
	}

	index_t* _stride = pascal_tensor_utils_default_stride(shape, ndim);

	tensor->size     = size;
	tensor->ndim     = ndim;
	tensor->shape    = _shape;
	tensor->_stride  = _stride;
	tensor->values   = values;

	return tensor;
}
