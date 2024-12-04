#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_tile(Tensor a, index_t tile_map[]) {
	index_t* shape = malloc(sizeof(index_t) * a->ndim);
	for (int i = 0; i < a->ndim; i++) {
		pascal_tensor_assert(tile_map[i] > 0, "Tile map accepts integers > 0\n");
		shape[i] = a->shape[i] * tile_map[i];
	}

	index_t  size    = pascal_tensor_utils_size_from_shape(shape, a->ndim);
	index_t* stride  = pascal_tensor_utils_default_stride(shape, a->ndim);

	double* values   = malloc(size * sizeof(double));

	index_t* indexes = malloc(a->ndim * sizeof(index_t));
	for (int i = 0; i < size; i++) {
		pascal_tensor_utils_index_from_linear_index(indexes, i, stride, a->ndim);
		for (int j = 0; j < a->ndim; j++) {
			indexes[j] %= a->shape[j];
		}

		values[i] = pascal_tensor_get(a, indexes);
	}
	free(indexes);

	Tensor tensor   = pascal_tensor_init();

	tensor->size    = size;
	tensor->ndim    = a->ndim;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->values  = values;

	return tensor;
}
