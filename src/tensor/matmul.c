#include <stdio.h>

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

Tensor pascal_tensor_matmul(Tensor a, Tensor b) {
	pascal_tensor_assert(a->ndim >= 2 && b->ndim >= 2, "Matmul requires at least 2 dimensions\n");
	pascal_tensor_assert(a->shape[a->ndim - 1] == b->shape[b->ndim - 2], "Shapes aren't compatible for matrix multiplication\n");

	index_t out_shape[2] = {a->shape[a->ndim - 2], b->shape[b->ndim - 1]};

	if (a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	if (b->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(b);
	}

	BroadcastOutput b_output      = pascal_tensor_broadcast_linalg(a, b, out_shape, 2);

	Tensor  c                     = b_output->tensor;
	double* values                = malloc(sizeof(double) * c->size);

	const index_t ROWS_A          = c->shape[c->ndim - 2];
	const index_t COLS_B          = c->shape[c->ndim - 1];
	const index_t COMM            = a->shape[a->ndim - 1];

	const index_t fixed_axis_size = ROWS_A * COLS_B;

	index_t* indexes              = malloc(sizeof(index_t) * (c->ndim - 2));
	for (int i = 0; i < c->ndim - 2; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < c->size / fixed_axis_size; i++) {

		index_t a_index = pascal_tensor_linear_index_from_index(indexes, b_output->a_stride, c->ndim - 2);
		index_t b_index = pascal_tensor_linear_index_from_index(indexes, b_output->b_stride, c->ndim - 2);

		double* _a      = a->values + a_index;
		double* _b      = b->values + b_index;

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
		const CBLAS_TRANSPOSE_t TRANSA = CblasNoTrans;
		const CBLAS_TRANSPOSE_t TRANSB = CblasNoTrans;

		gsl_matrix_view a              = gsl_matrix_view_array(_a, ROWS_A, COMM);
		gsl_matrix_view b              = gsl_matrix_view_array(_b, COMM, COLS_B);
		gsl_matrix*     _c             = gsl_matrix_alloc(ROWS_A, COLS_B);

		gsl_blas_dgemm(TRANSA, TRANSB, 1.0, &a.matrix, &b.matrix, 0.0, _c);

		for (int j = 0; j < ROWS_A; j++) {
			for (int k = 0; k < COLS_B; k++) {
				values[(i * fixed_axis_size) + (j * COLS_B + k)] = gsl_matrix_get(_c, j, k);
			}
		}

#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
		index_t                    values_offset = i * fixed_axis_size;
		const enum CBLAS_ORDER     ORDER         = CblasRowMajor;
		const enum CBLAS_TRANSPOSE TRANSA        = CblasNoTrans;
		const enum CBLAS_TRANSPOSE TRANSB        = CblasNoTrans;

		cblas_dgemm(ORDER, TRANSA, TRANSB, ROWS_A, COLS_B, COMM, 1.0, _a, COMM, _b, COLS_B, 0.0, values + values_offset, COLS_B);

#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
		index_t values_offset = i * fixed_axis_size;

		char TRANSA           = 'N';
		char TRANSB           = 'N';

		double alpha          = 1.0;
		double beta           = 0.0;

		double __a[ROWS_A * COMM];
		double __b[COMM * COLS_B];
		double __c[ROWS_A * COLS_B];

		for (int j = 0; j < ROWS_A; j++) {
			for (int k = 0; k < COMM; k++) {
				__a[j + k * ROWS_A] = _a[j * COMM + k];
			}
		}

		for (int j = 0; j < COMM; j++) {
			for (int k = 0; k < COLS_B; k++) {
				__b[j + k * COMM] = _b[j * COLS_B + k];
			}
		}

		int _ROWS_A = ROWS_A;
		int _COLS_B = COLS_B;
		int _COMM   = COMM;

		dgemm_(&TRANSA, &TRANSB, &_ROWS_A, &_COLS_B, &_COMM, &alpha, __a, &_ROWS_A, __b, &_COMM, &beta, __c, &_ROWS_A);

		for (int j = 0; j < ROWS_A; j++) {
			for (int k = 0; k < COLS_B; k++) {
				(values + values_offset)[j * COLS_B + k] = __c[j + k * ROWS_A];
			}
		}
#endif
		pascal_tensor_iterate_indexes_next(indexes, c->shape, c->ndim - 2);
	}
	free(indexes);

	c->values = values;

	pascal_tensor_broadcast_output_free(b_output);
	return c;
}
