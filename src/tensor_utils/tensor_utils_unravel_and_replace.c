#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

void pascal_tensor_utils_unravel_and_replace(Tensor a) {
	if (a->_transpose_map == NULL || a->_transpose_map_inverse == NULL) {
		return;
	}

	index_t* stride    = pascal_tensor_utils_default_stride(a->shape, a->ndim);

	double*  values    = malloc(a->size * sizeof(double));
	index_t* a_indexes = malloc(a->ndim * sizeof(index_t));
	for (int i = 0; i < a->size; i++) {
		pascal_tensor_utils_index_from_linear_index(a_indexes, i, stride, a->ndim);

		values[i] = pascal_tensor_get(a, a_indexes);
	}
	free(a_indexes);

	free(a->_stride);
	free(a->values);
	free(a->_transpose_map);
	free(a->_transpose_map_inverse);

	a->_stride                = stride;
	a->values                 = values;
	a->_transpose_map         = NULL;
	a->_transpose_map_inverse = NULL;
}
