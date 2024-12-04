#include "pascal.h"

Tensor pascal_tensor_utils_unary_chain_rule(Tensor current_grad, Tensor a, double (*gradient_fn)(double)) {
	Tensor tensor   = pascal_tensor_init();

	index_t  size   = a->size;
	index_t  ndim   = a->ndim;
	index_t* shape  = malloc(sizeof(index_t) * ndim);
	index_t* stride = malloc(ndim * sizeof(index_t));
	for (int i = 0; i < ndim; i++) {
		shape[i]  = a->shape[i];
		stride[i] = a->_stride[i];
	}

	double* values = malloc(sizeof(double) * size);
	for (int i = 0; i < size; i++) {
		values[i] = current_grad->values[i] * gradient_fn(a->values[i]);
	}

	tensor->size                   = size;
	tensor->ndim                   = ndim;
	tensor->shape                  = shape;
	tensor->_stride                = stride;
	tensor->values                 = values;
	tensor->_transpose_map         = a->_transpose_map;
	tensor->_transpose_map_inverse = a->_transpose_map_inverse;

	return tensor;
}
