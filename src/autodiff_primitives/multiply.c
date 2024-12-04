#include <stdlib.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_multiply_forward(Tensor* inputs) {
	return pascal_tensor_multiply(inputs[0], inputs[1]);
}

Tensor _autodiff_primitive_multiply_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 2, "Index must be less than 2");

	if (pascal_tensor_broadcast_is_needed(inputs[0], inputs[1])) {
		Tensor          t        = pascal_tensor_init();
		BroadcastOutput b_output = pascal_tensor_broadcast(inputs[0], inputs[1]);

		index_t* _b_stride;
		if (index == 0) {
			_b_stride = b_output->a_stride;
		} else {
			_b_stride = b_output->b_stride;
		}

		index_t  ndim  = inputs[index]->ndim;
		index_t  size  = inputs[index]->size;
		index_t* shape = malloc(ndim * sizeof(index_t));
		for (index_t i = 0; i < ndim; i++) {
			shape[i] = inputs[index]->shape[i];
		}
		index_t* stride = pascal_tensor_utils_default_stride(shape, ndim);

		double* values  = malloc(size * sizeof(double));
		for (index_t i = 0; i < size; i++) {
			values[i] = 0;
		}

		index_t* indexes = malloc(forward->ndim * sizeof(index_t));
		for (index_t i = 0; i < forward->ndim; i++) {
			indexes[i] = 0;
		}

		for (index_t i = 0; i < forward->size; i++) {
			index_t current_index = pascal_tensor_linear_index_from_index(indexes, _b_stride, forward->ndim);

			values[current_index] += forward->values[i] / inputs[index]->values[current_index] * current_grad->values[i];

			pascal_tensor_iterate_indexes_next(indexes, forward->shape, forward->ndim);
		}
		free(indexes);

		t->ndim    = ndim;
		t->size    = size;
		t->shape   = shape;
		t->_stride = stride;
		t->values  = values;

		pascal_tensor_free(b_output->tensor);
		pascal_tensor_broadcast_output_free(b_output);

		return t;
	}

	return pascal_tensor_multiply(inputs[(index + 1) % 2], current_grad);
}
