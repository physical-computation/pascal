#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

index_t* pascal_tensor_utils_default_stride(index_t shape[], index_t ndim) {
	index_t* stride  = malloc(sizeof(index_t) * ndim);

	stride[ndim - 1] = 1;
	for (int i = ndim - 2; i >= 0; i--) {
		stride[i] = stride[i + 1] * shape[i + 1];
	}

	return stride;
}
