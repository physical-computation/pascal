#include "pascal_autodiff.h"

AutodiffNode pascal_autodiff_parameter(Tensor value) {
	AutodiffNode new_node                = pascal_autodiff_init();

	new_node->is_parameter               = true;
	new_node->_is_necessary_for_gradient = true;

	new_node->forward                    = value;
	new_node->operation                  = "parameter";

	return new_node;
}
