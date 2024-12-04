#include <stdlib.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_add_forward(Tensor* inputs) {
	return pascal_tensor_add(inputs[0], inputs[1]);
}

Tensor _autodiff_primitive_add_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 2, "Index must be less than 2");

	if (!pascal_tensor_broadcast_is_needed(current_grad, inputs[index])) {
		return pascal_tensor_copy(current_grad);
	}

	bool* sum_mask = malloc(current_grad->ndim * sizeof(bool));
	for (int i = 0; i < inputs[index]->ndim; i++) {
		int current_i       = current_grad->ndim - 1 - i;
		int input_i         = inputs[index]->ndim - 1 - i;
		sum_mask[current_i] = (inputs[index]->shape[input_i] != current_grad->shape[current_i]);
	}

	for (int i = 0; i < current_grad->ndim - inputs[index]->ndim; i++) {
		sum_mask[i] = true;
	}

	Tensor out_sum = pascal_tensor_sum_mask(current_grad, sum_mask);
	Tensor out     = pascal_tensor_reshape(out_sum, inputs[index]->shape, inputs[index]->ndim);

	free(sum_mask);
	pascal_tensor_free(out_sum);

	return out;
}
