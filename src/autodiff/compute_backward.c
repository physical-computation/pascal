#include <stdlib.h>

#include "pascal_autodiff.h"
#include "pascal.h"

static Tensor map_operation_gradient(Tensor* inputs, double (*map_gradient)(double), Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");

	return pascal_tensor_self_derivative_with_operation(inputs[index], current_grad, map_gradient);
}

static double clamp_operation_gradient(double x, double clamp_min, double clamp_max) {
	if (x < clamp_min || x > clamp_max) {
		return 0;
	} else {
		return 1;
	}
}

static Tensor clamp_gradient(Tensor* inputs, double clamp_min, double clamp_max, Tensor current_grad, index_t index) {
	pascal_tensor_assert(index < 1, "Index must be less than 1");
	Tensor gradient = pascal_tensor_init();

	index_t  ndim   = inputs[index]->ndim;
	index_t  size   = inputs[index]->size;
	index_t* shape  = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		shape[i] = inputs[index]->shape[i];
	}

	index_t* stride = pascal_tensor_utils_default_stride(shape, ndim);

	double* values  = malloc(sizeof(double) * size);
	for (int i = 0; i < size; i++) {
		values[i] = clamp_operation_gradient(inputs[index]->values[i], clamp_min, clamp_max) * current_grad->values[i];
	}

	gradient->ndim    = ndim;
	gradient->size    = size;
	gradient->shape   = shape;
	gradient->_stride = stride;
	gradient->values  = values;

	return gradient;
}

static Tensor compute_chain_rule(AutodiffNode node, Tensor* inputs, Tensor current_grad, index_t index) {
	Tensor t;
	if (node->_transform_info->map) {
		t = map_operation_gradient(inputs, node->_transform_info->map_gradient, current_grad, index);
	} else if (node->_transform_info->clamp) {
		t = clamp_gradient(inputs, node->_transform_info->clamp_min, node->_transform_info->clamp_max, current_grad, index);
	} else {
		t = node->gradient_fn(inputs, node->forward, current_grad, index);
	}

	return t;
}

static void backward_recursion(AutodiffNode node, Tensor current_derivative) {
	if (node->grad == NULL) {
		node->grad = pascal_tensor_copy(current_derivative);
	} else {
		Tensor _gradient = pascal_tensor_add(node->grad, current_derivative);

		pascal_tensor_free(node->grad);
		node->grad = _gradient;
	}

	if (node->num_inputs <= 0) {
		return;
	}

	index_t num_inputs = node->num_inputs;
	Tensor* inputs     = malloc(num_inputs * sizeof(Tensor));
	for (int i = 0; i < num_inputs; i++) {
		inputs[i] = node->next[i]->forward;
	}

	for (int i = 0; i < num_inputs; i++) {
		if (!node->next[i]->_is_necessary_for_gradient) {
			continue;
		}

		Tensor new_derivative = compute_chain_rule(node, inputs, current_derivative, i);

		backward_recursion(node->next[i], new_derivative);
		pascal_tensor_free(new_derivative);
	}
	free(inputs);
}

void pascal_autodiff_compute_backward(AutodiffNode node) {
	Tensor current_derivative = pascal_tensor_self_derivative(node->forward);
	backward_recursion(node, current_derivative);

	pascal_tensor_free(current_derivative);
}
