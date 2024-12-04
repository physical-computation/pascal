#include <stdlib.h>

#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"
#include "pascal.h"

AutodiffNode pascal_autodiff_map(AutodiffNode node, double (*map_forward)(double), double (*map_gradient)(double)) {

	AutodiffNode new_node                   = pascal_autodiff_init();
	new_node->operation                     = "map";

	new_node->num_inputs                    = 1;

	new_node->next                          = malloc(sizeof(AutodiffNode));

	new_node->next[0]                       = node;
	new_node->_is_necessary_for_gradient    = node->_is_necessary_for_gradient;

	new_node->forward                       = NULL;

	new_node->_transform_info->map_forward  = map_forward;
	new_node->_transform_info->map_gradient = map_gradient;
	new_node->_transform_info->map          = true;

	Tensor inputs[1]                        = {node->forward};

	return new_node;
}
