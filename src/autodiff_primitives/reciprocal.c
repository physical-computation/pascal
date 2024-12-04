#include <math.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

static double reciprocal_grad(double x) {
	return -1.0 * pow(x * x, -1);
}

Tensor _autodiff_primitive_reciprocal_forward(Tensor* inputs) {
	return pascal_tensor_reciprocal(inputs[0]);
}

Tensor _autodiff_primitive_reciprocal_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	return pascal_tensor_utils_unary_chain_rule(current_grad, inputs[0], reciprocal_grad);
}
