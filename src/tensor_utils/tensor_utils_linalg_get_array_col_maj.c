#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

double* pascal_tensor_utils_linalg_get_array_col_maj(Tensor a, index_t indexes[]) {
	index_t ndim_1    = a->shape[a->ndim - 2];
	index_t ndim_2    = a->shape[a->ndim - 1];

	double*  values   = malloc((ndim_1 * ndim_2) * sizeof(double));
	index_t* _indexes = malloc(a->ndim * sizeof(index_t));

	for (int i = 0; i < a->ndim; i++) {
		_indexes[i] = indexes[i];
	}

	for (int i = 0; i < ndim_1; i++) {
		indexes[a->ndim - 2] = _indexes[a->ndim - 2] + i;
		for (int j = 0; j < ndim_2; j++) {
			index_t index        = j * (ndim_1) + i;
			indexes[a->ndim - 1] = _indexes[a->ndim - 1] + j;

			values[index]        = pascal_tensor_get(a, indexes);
		}
	}

	free(_indexes);
	return values;
}
