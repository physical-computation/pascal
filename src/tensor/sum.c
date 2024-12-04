#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static Tensor pascal_tensor_sum_axes_and_mask(Tensor a, index_t axes[], bool axes_mask[], index_t n_axes) {
	Tensor tensor = pascal_tensor_init();

	if (a->_transpose_map != NULL || a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	index_t  ndim  = a->ndim;
	index_t* shape = malloc((ndim) * sizeof(index_t));

	for (int i = 0; i < a->ndim; i++) {
		if (axes_mask[i]) {
			shape[i] = 1;
			continue;
		}
		shape[i] = a->shape[i];
	}

	double* values = malloc(a->size * sizeof(double));
	for (int i = 0; i < a->size; i++) {
		values[i] = a->values[i];
	}

	index_t full_size = a->size;
	for (int it = 0; it < n_axes; it++) {
		index_t i            = axes[it];
		index_t sub_size     = a->_stride[i];

		index_t offset       = 0;

		index_t inner_offset = 0;
		index_t shape_i      = 0;
		for (int j = 0; j < full_size / sub_size; j++) {
			if (shape_i == a->shape[i]) {
				shape_i = 0;
				inner_offset += sub_size;
			}

			if (shape_i == 0) {
				for (int k = 0; k < sub_size; k++) {
					values[inner_offset + k] = values[offset + k];
				}
			} else {
				for (int k = 0; k < sub_size; k++) {
					values[inner_offset + k] += values[offset + k];
				}
			}

			shape_i++;
			offset += sub_size;
		}

		full_size /= a->shape[i];
	}

	index_t  sq_ndim  = ndim - n_axes;
	index_t* sq_shape = malloc(sq_ndim * sizeof(index_t));

	index_t sq_i      = 0;
	for (int i = 0; i < a->ndim; i++) {
		if (axes_mask[i]) {
			continue;
		}
		sq_shape[sq_i] = a->shape[i];
		sq_i++;
	}

	index_t  size      = pascal_tensor_utils_size_from_shape(sq_shape, sq_ndim);
	index_t* sq_stride = pascal_tensor_utils_default_stride(sq_shape, sq_ndim);

	values             = realloc(values, size * sizeof(double));

	tensor->size       = size;
	tensor->ndim       = sq_ndim;
	tensor->shape      = sq_shape;
	tensor->_stride    = sq_stride;
	tensor->values     = values;

	free(shape);
	return tensor;
}

Tensor pascal_tensor_sum(Tensor a, index_t axes[], index_t n_axes) {
	bool* axes_mask = malloc(a->ndim * sizeof(bool));
	for (int i = 0; i < a->ndim; i++) {
		axes_mask[i] = false;
	}

	for (int i = 0; i < n_axes; i++) {
		axes_mask[axes[i]] = true;
	}

	Tensor tensor = pascal_tensor_sum_axes_and_mask(a, axes, axes_mask, n_axes);

	free(axes_mask);
	return tensor;
}

Tensor pascal_tensor_sum_mask(Tensor a, bool axes_mask[]) {
	index_t n_axes = 0;
	for (int i = 0; i < a->ndim; i++) {
		if (axes_mask[i]) {
			n_axes++;
		}
	}

	index_t* axes   = malloc(n_axes * sizeof(index_t));
	index_t  axes_i = 0;
	for (int i = 0; i < a->ndim; i++) {
		if (axes_mask[i]) {
			axes[axes_i] = i;
			axes_i++;
		} else {
			axes_mask[i] = 0;
		}
	}

	Tensor tensor = pascal_tensor_sum_axes_and_mask(a, axes, axes_mask, n_axes);
	free(axes);

	return tensor;
}
