#include <stdlib.h>

#include "pascal_autodiff.h"
#include "pascal.h"

static Tensor map_operation_forward(Tensor* inputs, double (*map_forward)(double)) {
	return pascal_tensor_map(inputs[0], map_forward);
}

static Tensor clamp_forward(Tensor* inputs, double clamp_min, double clamp_max) {
	return pascal_tensor_clamp(inputs[0], clamp_min, clamp_max);
}

void pascal_autodiff_compute_forward(AutodiffNode node) {
	if (node->grad != NULL) {
		pascal_tensor_free(node->grad);
		node->grad = NULL;
	}

	if (node->next == NULL) {
		return;
	}

	if (node->forward != NULL) {
		pascal_tensor_free(node->forward);
		node->forward = NULL;
	}

	Tensor* inputs = malloc(node->num_inputs * sizeof(Tensor));
	for (int i = 0; i < node->num_inputs; i++) {
		pascal_autodiff_compute_forward(node->next[i]);
		inputs[i] = node->next[i]->forward;
	}

	if (node->_transform_info->map) {
		node->forward = map_operation_forward(inputs, node->_transform_info->map_forward);
	} else if (node->_transform_info->clamp) {
		node->forward = clamp_forward(inputs, node->_transform_info->clamp_min, node->_transform_info->clamp_max);
	} else {
		node->forward = node->forward_fn(inputs);
	}

	free(inputs);
}
