#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_prod_all_forward(Tensor* inputs) {
	return pascal_tensor_prod_all(inputs[0]);
}

Tensor _autodiff_primitive_prod_all_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	Tensor divided = pascal_tensor_divide(forward, inputs[index]);
	Tensor out     = pascal_tensor_multiply(divided, current_grad);

	pascal_tensor_free(divided);
	return out;
}
