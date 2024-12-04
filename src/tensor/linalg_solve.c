#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
	#include <gsl/gsl_blas.h>
	#include <gsl/gsl_linalg.h>
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
	#include <cblas.h>
	#ifdef VALGRIND
		#include <lapacke.h>
	#else
		#ifdef DARWIN
			#include <clapack.h>
		#else
			#include <lapack.h>
		#endif
	#endif
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
	#include "clapack/f2c.h"
	#include "clapack/blaswrap.h"
	#include "clapack/clapack.h"
#endif

Tensor pascal_tensor_linalg_solve(Tensor a, Tensor y) {
#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
	return pascal_tensor_matmul(pascal_tensor_linalg_inv(a), y);
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
	pascal_tensor_assert(a->shape[a->ndim - 1] == a->shape[a->ndim - 2], "Tensor a must be symmetric in the last 2 dimensions.\n");
	int M = a->shape[a->ndim - 1];

	pascal_tensor_assert(y->shape[y->ndim - 2] == M, "Tensor y must have the same number of rows as tensor a.\n");
	int K                         = y->shape[y->ndim - 1];

	index_t         out_ndim      = 2;
	index_t         out_shape[2]  = {M, K};
	BroadcastOutput b_output      = pascal_tensor_broadcast_linalg(a, y, out_shape, out_ndim);
	Tensor          x             = b_output->tensor;

	double*       values          = malloc(sizeof(double) * x->size);
	const index_t fixed_axis_size = M * K;

	int ipiv[M];
	int info;

	index_t* indexes = malloc(sizeof(index_t) * x->ndim);
	for (int i = 0; i < x->ndim; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < x->size / fixed_axis_size; i++) {
		index_t* masked_indexes_a = pascal_tensor_utils_get_masked_index(indexes, a->shape, a->ndim, x->ndim);
		index_t* masked_indexes_y = pascal_tensor_utils_get_masked_index(indexes, y->shape, y->ndim, x->ndim);

		double* _a                = pascal_tensor_utils_linalg_get_array_col_maj(a, masked_indexes_a);
		double* _y                = pascal_tensor_utils_linalg_get_array_col_maj(y, masked_indexes_y);

		dgesv_(&M, &K, _a, &M, ipiv, _y, &M, &info);
		pascal_tensor_assert(info == 0, "Singular matrix\n");

		for (int j = 0; j < M; j++) {
			for (int k = 0; k < K; k++) {
				values[(i * fixed_axis_size) + (j * K + k)] = _y[j + k * M];
			}
		}

		pascal_tensor_iterate_indexes_next(indexes, x->shape, x->ndim - 2);

		free(masked_indexes_a);
		free(masked_indexes_y);
		free(_a);
		free(_y);
	}
	free(indexes);

	x->values = values;

	pascal_tensor_broadcast_output_free(b_output);
	return x;
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
	pascal_tensor_assert(a->shape[a->ndim - 1] == a->shape[a->ndim - 2], "Tensor a must be symmetric in the last 2 dimensions.\n");
	int M = a->shape[a->ndim - 1];

	pascal_tensor_assert(y->shape[y->ndim - 2] == M, "Tensor y must have the same number of rows as tensor a.\n");
	int K                         = y->shape[y->ndim - 1];

	index_t         out_ndim      = 2;
	index_t         out_shape[2]  = {M, K};
	BroadcastOutput b_output      = pascal_tensor_broadcast_linalg(a, y, out_shape, out_ndim);
	Tensor          x             = b_output->tensor;

	double*       values          = malloc(sizeof(double) * x->size);
	const index_t fixed_axis_size = M * K;

	int ipiv[M];
	int info;

	index_t* indexes = malloc(sizeof(index_t) * x->ndim);
	for (int i = 0; i < x->ndim; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < x->size / fixed_axis_size; i++) {
		index_t* masked_indexes_a = pascal_tensor_utils_get_masked_index(indexes, a->shape, a->ndim, x->ndim);
		index_t* masked_indexes_y = pascal_tensor_utils_get_masked_index(indexes, y->shape, y->ndim, x->ndim);

		double* _a                = pascal_tensor_utils_linalg_get_array_col_maj(a, masked_indexes_a);
		double* _y                = pascal_tensor_utils_linalg_get_array_col_maj(y, masked_indexes_y);

		dgesv_(&M, &K, _a, &M, ipiv, _y, &M, &info);
		pascal_tensor_assert(info == 0, "Singular matrix\n");

		for (int j = 0; j < M; j++) {
			for (int k = 0; k < K; k++) {
				values[(i * fixed_axis_size) + (j * K + k)] = _y[j + k * M];
			}
		}

		pascal_tensor_iterate_indexes_next(indexes, x->shape, x->ndim - 2);

		free(masked_indexes_a);
		free(masked_indexes_y);
		free(_a);
		free(_y);
	}
	free(indexes);

	x->values = values;

	pascal_tensor_broadcast_output_free(b_output);
	return x;
#endif
}
