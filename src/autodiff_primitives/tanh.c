#include <math.h>

#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

static double _tanh(double x) {
	return (double)tanhl((long double)x);
}
static double tanh_grad(double x) {
	return 1 - (_tanh(x) * _tanh(x));
}

Tensor _autodiff_primitive_tanh_forward(Tensor* inputs) {
	return pascal_tensor_map(inputs[0], _tanh);
}

Tensor _autodiff_primitive_tanh_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	return pascal_tensor_utils_unary_chain_rule(current_grad, inputs[0], tanh_grad);
}
