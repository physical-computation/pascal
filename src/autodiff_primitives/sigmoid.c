#include <math.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

static double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

static double sigmoid_grad(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

Tensor _autodiff_primitive_sigmoid_forward(Tensor* inputs) {
	return pascal_tensor_map(inputs[0], sigmoid);
}

Tensor _autodiff_primitive_sigmoid_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	return pascal_tensor_utils_unary_chain_rule(current_grad, inputs[0], sigmoid_grad);
}
