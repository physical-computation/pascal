#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

index_t* pascal_tensor_utils_get_masked_index(index_t indexes[], index_t shape[], index_t ndim, index_t broadcasted_ndim) {
	index_t* return_indexes = malloc(ndim * sizeof(index_t));

	index_t broadcasted_i   = broadcasted_ndim - 1;
	for (int i = ndim - 1; i >= 0; i--) {
		return_indexes[i] = indexes[broadcasted_i] % shape[i];

		broadcasted_i--;
	}

	return return_indexes;
}
