#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_mean_all_forward(Tensor* inputs) {
	return pascal_tensor_mean_all(inputs[0]);
}

Tensor _autodiff_primitive_mean_all_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	double repeated_value = 1.0 / (double)inputs[index]->size;
	Tensor repeated       = pascal_tensor_new_repeat(repeated_value, inputs[0]->shape, inputs[0]->ndim);
	Tensor out            = pascal_tensor_multiply(repeated, current_grad);

	pascal_tensor_free(repeated);
	return out;
}
