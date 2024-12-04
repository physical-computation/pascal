#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

index_t pascal_tensor_utils_size_from_shape(index_t shape[], index_t ndim) {
	index_t size = 1;

	for (int i = 0; i < ndim; i++) {
		size *= shape[i];
	}

	return size;
}
