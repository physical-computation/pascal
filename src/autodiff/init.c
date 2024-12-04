#include <stdlib.h>

#include "pascal_autodiff.h"

AutodiffNode pascal_autodiff_init() {
	AutodiffNode new_node                = malloc(sizeof(pascal_autodiff_D));

	new_node->operation                  = NULL;
	new_node->num_inputs                 = 0;
	new_node->is_parameter               = false;
	new_node->_is_necessary_for_gradient = false;
	new_node->forward_fn                 = NULL;
	new_node->gradient_fn                = NULL;
	new_node->chain_rule_fn              = NULL;
	new_node->next                       = NULL;

	new_node->grad                       = NULL;
	new_node->forward                    = NULL;

	TransformInfo _transform_info        = malloc(sizeof(TransformInfo_D));
	_transform_info->map_forward         = NULL;
	_transform_info->map_gradient        = NULL;
	_transform_info->map                 = false;
	_transform_info->clamp_min           = 0;
	_transform_info->clamp_max           = 0;
	_transform_info->clamp               = false;

	new_node->_transform_info            = _transform_info;

	return new_node;
}
