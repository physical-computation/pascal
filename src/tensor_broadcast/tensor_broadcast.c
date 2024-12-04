
#include <stdlib.h>

#include "pascal.h"

BroadcastOutput pascal_tensor_broadcast(Tensor a, Tensor b) {
	BroadcastOutput b_output = pascal_tensor_broadcast_output_init();

	index_t* shape;
	index_t  ndim;

	Tensor tensor = pascal_tensor_init();

	if (a->ndim >= b->ndim) {
		shape = malloc(sizeof(index_t) * a->ndim);
		ndim  = a->ndim;
	} else {
		shape = malloc(sizeof(index_t) * b->ndim);
		ndim  = b->ndim;
	}

	index_t* a_stride = malloc(sizeof(index_t) * ndim);
	index_t* b_stride = malloc(sizeof(index_t) * ndim);

	int a_counter     = a->ndim - 1;
	int b_counter     = b->ndim - 1;
	int s_counter     = ndim - 1;

	while (s_counter >= 0) {
		if (a_counter >= 0 && b_counter >= 0) {
			index_t a_shape = a->shape[a_counter];
			index_t b_shape = b->shape[b_counter];

			pascal_tensor_assert(a_shape == b_shape || a_shape == 1 || b_shape == 1, "Shapes aren't compatible for broadcasting.");
			if (a_shape > b_shape) {
				shape[s_counter]    = a_shape;
				a_stride[s_counter] = a->_stride[a_counter];
				b_stride[s_counter] = 0;
			} else if (a_shape == b_shape) {
				shape[s_counter]    = a_shape;
				a_stride[s_counter] = a->_stride[a_counter];
				b_stride[s_counter] = b->_stride[b_counter];
			} else {
				shape[s_counter]    = b_shape;
				a_stride[s_counter] = 0;
				b_stride[s_counter] = b->_stride[b_counter];
			}
		} else if (a_counter >= 0) {
			shape[s_counter]    = a->shape[a_counter];
			a_stride[s_counter] = a->_stride[a_counter];
			b_stride[s_counter] = 0;
		} else {
			shape[s_counter]    = b->shape[b_counter];
			a_stride[s_counter] = 0;
			b_stride[s_counter] = b->_stride[b_counter];
		}

		a_counter--;
		b_counter--;
		s_counter--;
	}

	index_t size       = pascal_tensor_utils_size_from_shape(shape, ndim);

	tensor->size       = size;
	tensor->ndim       = ndim;
	tensor->shape      = shape;
	tensor->_stride    = pascal_tensor_utils_default_stride(shape, ndim);

	b_output->tensor   = tensor;
	b_output->a_stride = a_stride;
	b_output->b_stride = b_stride;

	return b_output;
}
