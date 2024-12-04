#include "pascal_autodiff.h"

void pascal_autodiff_print(AutodiffNode node) {
	pascal_autodiff_compute_forward(node);
	if (node->forward != NULL) {
		pascal_tensor_print(node->forward);
	}
}
