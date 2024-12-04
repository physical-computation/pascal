#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

static double relu(double x) {
	return x > 0 ? x : 0;
}

static double relu_grad(double x) {
	return x > 0 ? 1 : 0;
}

Tensor _autodiff_primitive_relu_forward(Tensor* inputs) {
	return pascal_tensor_map(inputs[0], relu);
}

Tensor _autodiff_primitive_relu_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	return pascal_tensor_utils_unary_chain_rule(current_grad, inputs[0], relu_grad);
}
