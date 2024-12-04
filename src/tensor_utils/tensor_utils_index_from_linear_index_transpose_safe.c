#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

index_t* pascal_tensor_utils_index_from_linear_index_transpose_safe(index_t linear_index, Tensor a) {
	index_t* stride       = a->_stride;
	index_t  ndim         = a->ndim;
	index_t  _rolling_sum = 0;

	if (a->_transpose_map == NULL) {
		index_t* indexes = malloc(ndim * sizeof(index_t));

		for (int i = 0; i < ndim; i++) {
			indexes[i] = (linear_index - _rolling_sum) / stride[i];
			_rolling_sum += indexes[i] * stride[i];
		}

		return indexes;

	} else {
		index_t* _indexes          = malloc(ndim * sizeof(index_t));

		index_t* _corrected_stride = pascal_tensor_utils_apply_transpose_map(stride, a->_transpose_map_inverse, ndim);
		for (int i = 0; i < ndim; i++) {
			_indexes[i] = (linear_index - _rolling_sum) / _corrected_stride[i];
			_rolling_sum += _indexes[i] * _corrected_stride[i];
		}

		index_t* corrected_indexes = pascal_tensor_utils_apply_transpose_map(_indexes, a->_transpose_map, ndim);

		free(_corrected_stride);
		free(_indexes);

		return corrected_indexes;
	}
}
