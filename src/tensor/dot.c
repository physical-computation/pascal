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

Tensor pascal_tensor_dot(Tensor a, Tensor b) {
	if (a->shape[a->ndim - 1] == 1 && b->shape[b->ndim - 1] == 1) {
		pascal_tensor_assert(a->shape[a->ndim - 2] == b->shape[b->ndim - 2], "Shapes aren't compatible for taking the dot product\n");
	} else if (a->shape[a->ndim - 2] == 1 && b->shape[b->ndim - 2] == 1) {
		pascal_tensor_assert(a->shape[a->ndim - 1] == b->shape[b->ndim - 1], "Shapes aren't compatible for taking the dot product\n");
	} else {
		pascal_tensor_assert(false, "Shapes aren't compatible for taking the dot product\n");
	}

	if (a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	if (b->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(b);
	}

	index_t         out_shape[2] = {1, 1};
	BroadcastOutput b_output     = pascal_tensor_broadcast_linalg(a, b, out_shape, 2);
	Tensor          c            = b_output->tensor;

	double* values               = malloc(sizeof(double) * c->size);

	const index_t ORDER          = a->shape[a->ndim - 2] * a->shape[a->ndim - 1];

	index_t* indexes             = malloc(sizeof(index_t) * (c->ndim - 2));
	for (int i = 0; i < c->ndim - 2; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < c->size; i++) {
		index_t a_index = pascal_tensor_linear_index_from_index(indexes, b_output->a_stride, c->ndim - 2);
		index_t b_index = pascal_tensor_linear_index_from_index(indexes, b_output->b_stride, c->ndim - 2);

		double* _a      = a->values + a_index;
		double* _b      = b->values + b_index;
#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
		double          result;
		gsl_vector_view a = gsl_vector_view_array(_a, ORDER);
		gsl_vector_view b = gsl_vector_view_array(_b, ORDER);
		gsl_blas_ddot(&a.vector, &b.vector, &result);

		values[i] = result;
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
		values[i] = cblas_ddot(ORDER, _a, 1, _b, 1);
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
		integer n    = ORDER;
		integer incx = 1;
		integer incy = 1;
		values[i]    = ddot_(&n, _a, &incx, _b, &incy);
#endif
		pascal_tensor_iterate_indexes_next(indexes, c->shape, c->ndim - 2);
	}
	free(indexes);

	c->values = values;

	pascal_tensor_broadcast_output_free(b_output);
	return c;
}
