#include <math.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

Tensor _autodiff_primitive_sin_forward(Tensor* inputs) {
	return pascal_tensor_map(inputs[0], sin);
}

Tensor _autodiff_primitive_sin_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	return pascal_tensor_utils_unary_chain_rule(current_grad, inputs[0], cos);
}
