#include "pascal_autodiff.h"

AutodiffNode pascal_autodiff_new(Tensor value) {
	AutodiffNode new_node = pascal_autodiff_init();

	new_node->operation   = "new";
	new_node->forward     = value;

	return new_node;
}
