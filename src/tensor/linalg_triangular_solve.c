#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

void _lower_triangular_solve(double* a, double* y, index_t M, index_t K, double* out) {
	for (index_t i = 0; i < M; i++) {
		for (index_t j = 0; j < K; j++) {
			double sum = 0;
			for (index_t k = 0; k < i; k++) {
				sum += a[i * M + k] * out[k * K + j];
			}

			out[i * K + j] = (y[i * K + j] - sum) / a[i * M + i];
		}
	}
}

void _upper_triangular_solve(double* a, double* y, index_t M, index_t K, double* out) {
	for (index_t i = M; i > 0; i--) {
		for (index_t j = 0; j < K; j++) {
			double sum = 0;
			for (index_t k = M; k > i; k--) {
				sum += a[(i - 1) * M + (k - 1)] * out[(k - 1) * K + j];
			}
			out[(i - 1) * K + j] = (y[(i - 1) * K + j] - sum) / a[(i - 1) * M + (i - 1)];
		}
	}
}

Tensor pascal_tensor_linalg_triangular_solve(Tensor a, Tensor y, bool lower) {
	pascal_tensor_assert(a->shape[a->ndim - 1] == a->shape[a->ndim - 2], "Tensor a must be symmetric in the last 2 dimensions.\n");
	int M = a->shape[a->ndim - 1];

	pascal_tensor_assert(y->shape[y->ndim - 2] == M, "Tensor y must have the same number of rows as tensor a.\n");
	int K = y->shape[y->ndim - 1];

	if (a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	if (y->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(y);
	}

	index_t         out_ndim      = 2;
	index_t         out_shape[2]  = {M, K};
	BroadcastOutput b_output      = pascal_tensor_broadcast_linalg(a, y, out_shape, out_ndim);
	Tensor          x             = b_output->tensor;

	double* values                = malloc(sizeof(double) * x->size);
	x->values                     = values;

	const index_t fixed_axis_size = M * K;

	index_t* indexes              = malloc(sizeof(index_t) * x->ndim);
	for (int i = 0; i < x->ndim; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < x->size / fixed_axis_size; i++) {
		index_t a_index      = pascal_tensor_linear_index_from_index(indexes, b_output->a_stride, x->ndim - 2);
		index_t y_index      = pascal_tensor_linear_index_from_index(indexes, b_output->b_stride, x->ndim - 2);
		index_t values_index = pascal_tensor_linear_index_from_index(indexes, x->_stride, x->ndim - 2);

		double* _a           = a->values + a_index;
		double* _y           = y->values + y_index;
		double* _values      = x->values + values_index;

		if (lower) {
			_lower_triangular_solve(_a, _y, M, K, _values);
		} else {
			_upper_triangular_solve(_a, _y, M, K, _values);
		}

		pascal_tensor_iterate_indexes_next(indexes, x->shape, x->ndim - 2);
	}
	free(indexes);
	pascal_tensor_broadcast_output_free(b_output);

	return x;
}
