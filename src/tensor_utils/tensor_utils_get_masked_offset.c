#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

index_t pascal_tensor_utils_get_masked_offset(index_t indexes[], Tensor a, index_t broadcasted_ndim) {
	index_t offset        = 0;

	index_t broadcasted_i = broadcasted_ndim - 1;
	for (int i = a->ndim - 1; i >= 0; i--) {
		offset += (indexes[broadcasted_i] % a->shape[i]) * a->_stride[i];

		broadcasted_i--;
	}

	return offset;
}
