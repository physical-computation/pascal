#include <stdlib.h>

#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

AutodiffNode pascal_autodiff_clamp(AutodiffNode node, double clamp_min, double clamp_max) {
	AutodiffNode new_node                = pascal_autodiff_init();

	new_node->num_inputs                 = 1;

	new_node->next                       = malloc(sizeof(AutodiffNode));
	new_node->next[0]                    = node;

	new_node->_is_necessary_for_gradient = node->_is_necessary_for_gradient;

	new_node->forward                    = NULL;

	new_node->_transform_info->clamp_min = clamp_min;
	new_node->_transform_info->clamp_max = clamp_max;
	new_node->_transform_info->clamp     = true;

	Tensor inputs[1]                     = {node->forward};

	return new_node;
}
