#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

index_t* pascal_tensor_utils_get_convolution_start_index(index_t indexes[], index_t stride[], index_t indexes_ndim, index_t stride_ndim) {
	index_t* start_index = malloc(sizeof(index_t) * indexes_ndim);

	for (int i = 0; i < indexes_ndim; i++) {
		if (i < indexes_ndim - stride_ndim) {
			start_index[i] = indexes[i];
		} else {
			index_t stride_i = i - (indexes_ndim - stride_ndim);
			if (stride[stride_i] == 0) {
				start_index[i] = 0;
			} else {
				start_index[i] = indexes[i] * stride[stride_i];
			}
		}
	}

	return start_index;
}
