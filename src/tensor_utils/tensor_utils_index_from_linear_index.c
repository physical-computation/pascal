#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

void pascal_tensor_utils_index_from_linear_index(index_t out[], index_t linear_index, index_t stride[], index_t ndim) {
	index_t _rolling_sum = 0;

	for (int i = 0; i < ndim; i++) {
		out[i] = (linear_index - _rolling_sum) / stride[i];
		_rolling_sum += out[i] * stride[i];
	}
}
