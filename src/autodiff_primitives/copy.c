#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_copy_forward(Tensor* inputs) {
	return pascal_tensor_copy(inputs[0]);
}

Tensor _autodiff_primitive_copy_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	return pascal_tensor_copy(current_grad);
}
