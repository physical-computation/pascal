#include <stdbool.h>

#include "pascal.h"

bool pascal_tensor_broadcast_is_needed(Tensor a, Tensor b) {
	if (a->ndim != b->ndim) {
		return true;
	}

	for (int i = 0; i < a->ndim; i++) {
		if (a->shape[i] != b->shape[i]) {
			return true;
		}
	}

	return false;
}
