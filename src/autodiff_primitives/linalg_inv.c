#include <stdlib.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_linalg_inv_forward(Tensor* inputs) {
	return pascal_tensor_linalg_inv(inputs[0]);
}

Tensor _autodiff_primitive_linalg_inv_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	Tensor self            = pascal_tensor_self_derivative(inputs[index]);
	Tensor neg_inv         = pascal_tensor_scalar_multiply(forward, -1);

	Tensor first_matmul    = pascal_tensor_matmul(self, forward);
	Tensor grad            = pascal_tensor_matmul(neg_inv, first_matmul);
	Tensor result_t        = pascal_tensor_multiply(grad, current_grad);

	index_t  ndim          = inputs[0]->ndim;
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

	Tensor result = pascal_tensor_transpose(result_t, transpose_map);

	pascal_tensor_free(self);
	pascal_tensor_free(neg_inv);
	pascal_tensor_free(first_matmul);
	pascal_tensor_free(grad);
	pascal_tensor_free(result_t);
	free(transpose_map);

	return result;
}
