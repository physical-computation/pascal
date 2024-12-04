#include <stdbool.h>

#include "pascal.h"

bool pascal_tensor_broadcast_is_needed_linalg(Tensor a, Tensor b) {
	if (a->ndim != b->ndim) {
		return true;
	}

	for (int i = 0; i < a->ndim - 2; i++) {
		if (a->shape[i] != b->shape[i]) {
			return true;
		}
	}

	return false;
}
