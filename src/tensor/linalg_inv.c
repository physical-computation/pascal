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

Tensor pascal_tensor_linalg_inv(Tensor a) {
	pascal_tensor_assert(a->shape[a->ndim - 1] == a->shape[a->ndim - 2], "The last 2 dimensions must be equal (square matrix)\n");

	if (a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	int M          = a->shape[a->ndim - 1];
	int ORDER      = M * M;

	double* values = malloc(a->size * sizeof(double));

	int    ipiv[M];
	double work[ORDER];
	int    info;

	index_t* indexes = malloc(sizeof(index_t) * a->ndim);
	for (int i = 0; i < a->ndim; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < a->size / ORDER; i++) {
		index_t* masked_indexes_a = pascal_tensor_utils_get_masked_index(indexes, a->shape, a->ndim, a->ndim);

		double* _a                = pascal_tensor_utils_linalg_get_array_col_maj(a, masked_indexes_a);

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
		gsl_matrix_view m = gsl_matrix_view_array(_a, M, M);

		int              s;
		gsl_permutation* p = gsl_permutation_alloc(M);

		gsl_linalg_LU_decomp(&m.matrix, p, &s);
		gsl_linalg_LU_invx(&m.matrix, p);
		gsl_permutation_free(p);

		for (int j = 0; j < M; j++) {
			for (int k = 0; k < M; k++) {
				values[(i * ORDER) + ((j * M) + k)] = gsl_matrix_get(&m.matrix, k, j);
			}
		}
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
		dgetrf_(&M, &M, _a, &M, ipiv, &info);
		pascal_tensor_assert(info == 0, "Singular matrix\n");

		dgetri_(&M, _a, &M, ipiv, work, &ORDER, &info);
		pascal_tensor_assert(info == 0, "Singular matrix\n");

		for (int j = 0; j < M; j++) {
			for (int k = 0; k < M; k++) {
				values[(i * ORDER) + ((j * M) + k)] = _a[j + (k * M)];
			}
		}
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
		dgetrf_(&M, &M, _a, &M, ipiv, &info);
		pascal_tensor_assert(info == 0, "Singular matrix\n");

		dgetri_(&M, _a, &M, ipiv, work, &ORDER, &info);
		pascal_tensor_assert(info == 0, "Singular matrix\n");

		for (int j = 0; j < M; j++) {
			for (int k = 0; k < M; k++) {
				values[(i * ORDER) + ((j * M) + k)] = _a[j + (k * M)];
			}
		}
#endif
		pascal_tensor_iterate_indexes_next(indexes, a->shape, a->ndim - 2);

		free(masked_indexes_a);
		free(_a);
	}
	free(indexes);

	Tensor a_inv = pascal_tensor_new_no_malloc(values, a->shape, a->ndim);
	return a_inv;
}
