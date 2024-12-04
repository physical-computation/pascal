#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

index_t* pascal_tensor_utils_apply_transpose_map(index_t* indexes, index_t* transpose_map, index_t ndim) {
	index_t* return_indexes = malloc(ndim * sizeof(index_t));

	for (int i = 0; i < ndim; i++) {
		return_indexes[i] = indexes[transpose_map[i]];
	}

	return return_indexes;
}
