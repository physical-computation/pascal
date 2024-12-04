#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_mean_all(Tensor a) {
	Tensor   tensor = pascal_tensor_init();
	index_t  ndim   = 1;
	index_t  size   = 1;
	index_t* shape  = malloc(ndim * sizeof(index_t));
	shape[0]        = 1;

	index_t* stride = malloc(ndim * sizeof(index_t));
	stride[0]       = 1;

	double* values  = malloc(size * sizeof(double));
	double  value   = 0.0;
	for (int i = 0; i < a->size; i++) {
		value += a->values[i];
	}
	values[0]       = value / a->size;

	tensor->size    = size;
	tensor->ndim    = ndim;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->values  = values;

	return tensor;
}
