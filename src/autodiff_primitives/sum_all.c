#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_sum_all_forward(Tensor* inputs) {
	return pascal_tensor_sum_all(inputs[0]);
}

Tensor _autodiff_primitive_sum_all_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	Tensor ones = pascal_tensor_ones(inputs[0]->shape, inputs[0]->ndim);
	Tensor out  = pascal_tensor_multiply(ones, current_grad);

	pascal_tensor_free(ones);
	return out;
}
