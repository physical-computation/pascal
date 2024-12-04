#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static double clamp_operation(double a, double min, double max) {
	if (a < min) {
		return min;
	} else if (a > max) {
		return max;
	} else {
		return a;
	}
}

Tensor pascal_tensor_clamp(Tensor a, double min, double max) {
	pascal_tensor_assert(max >= min, "Max must be greater than or equal to min\n");
	Tensor tensor = pascal_tensor_copy(a);

	for (int i = 0; i < tensor->size; i++) {
		tensor->values[i] = clamp_operation(tensor->values[i], min, max);
	}

	return tensor;
}
