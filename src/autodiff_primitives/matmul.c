#include <stdlib.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_matmul_forward(Tensor* inputs) {
	return pascal_tensor_matmul(inputs[0], inputs[1]);
}

Tensor _autodiff_primitive_matmul_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 2, "Index must be less than 2");

	index_t  swapped_index = (index + 1) % 2;
	index_t  ndim          = inputs[swapped_index]->ndim;
	index_t* transpose_map = malloc((ndim * sizeof(index_t)));
	for (int i = 0; i < ndim; i++) {
		if (i == ndim - 2) {
			transpose_map[i] = ndim - 1;
			continue;
		} else if (i == ndim - 1) {
			transpose_map[i] = ndim - 2;
			continue;
		}
		transpose_map[i] = i;
	}

	Tensor transposed = pascal_tensor_transpose(inputs[swapped_index], transpose_map);
	Tensor t;
	if (index == 0) {
		t = pascal_tensor_matmul(current_grad, transposed);
		pascal_tensor_free(transposed);
	} else {
		t = pascal_tensor_matmul(transposed, current_grad);
		pascal_tensor_free(transposed);
	}
	free(transpose_map);

	if (!pascal_tensor_broadcast_is_needed_linalg(current_grad, inputs[index])) {
		return t;
	}

	bool* sum_mask = calloc(current_grad->ndim, sizeof(bool));
	for (int i = 0; i < inputs[index]->ndim - 2; i++) {
		int current_i       = current_grad->ndim - 3 - i;
		int input_i         = inputs[index]->ndim - 3 - i;
		sum_mask[current_i] = (inputs[index]->shape[input_i] != current_grad->shape[current_i]);
	}

	for (int i = 0; i < current_grad->ndim - inputs[index]->ndim; i++) {
		sum_mask[i] = true;
	}

	sum_mask[inputs[index]->ndim - 2] = false;
	sum_mask[inputs[index]->ndim - 1] = false;

	Tensor out_sum                    = pascal_tensor_sum_mask(t, sum_mask);
	Tensor out                        = pascal_tensor_reshape(out_sum, inputs[index]->shape, inputs[index]->ndim);

	free(sum_mask);
	pascal_tensor_free(out_sum);

	pascal_tensor_free(t);

	return out;
}
