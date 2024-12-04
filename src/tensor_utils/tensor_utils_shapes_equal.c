#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

bool pascal_tensor_utils_shapes_equal(Tensor a, Tensor b) {

	bool shapes_equal = true;

	if (a->ndim != b->ndim) {
		shapes_equal = shapes_equal && false;
		return shapes_equal;
	}

	for (int i = 0; i < a->ndim; i++) {
		if (a->shape[i] != b->shape[i]) {
			shapes_equal = shapes_equal && false;
		}
	}

	return shapes_equal;
}
