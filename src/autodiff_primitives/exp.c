#include <math.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_exp_forward(Tensor* inputs) {
	return pascal_tensor_map(inputs[0], exp);
}

Tensor _autodiff_primitive_exp_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	return pascal_tensor_utils_unary_chain_rule(current_grad, inputs[0], exp);
}
