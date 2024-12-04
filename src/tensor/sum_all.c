#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_USE_SIMD
	#include <arm_neon.h>
#endif

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
#endif

#if TENSOR_USE_SIMD
static void simd_operation_sum_all(double* out, double* a, double* b) {
	float64x2_t v;

	float64x2_t _a = vld1q_f64(a);
	float64x2_t _b = vld1q_f64(b);

	v              = vaddq_f64(_a, _b);

	(*out) += vaddvq_f64(v);
}
#endif

Tensor pascal_tensor_sum_all(Tensor a) {
	Tensor   tensor = pascal_tensor_init();
	index_t  ndim   = 1;
	index_t  size   = 1;
	index_t* shape  = malloc(ndim * sizeof(index_t));
	shape[0]        = 1;

	index_t* stride = malloc(ndim * sizeof(index_t));
	stride[0]       = 1;

	// #if TENSOR_USE_SIMD
	//     double* values = malloc(size * sizeof(double));
	//     double value0 = 0;
	//     double value1 = 0;
	//     double value2 = 0;
	//     double value3 = 0;

	//     int i = 0;
	//     while (i < a->size) {
	//         if (a->size - i >= 16) {
	//             simd_operation_sum_all(&value0, a->values + i, a->values + i + 2);
	//             simd_operation_sum_all(&value1, a->values + i + 4, a->values + i + 6);
	//             simd_operation_sum_all(&value2, a->values + i + 8, a->values + i + 10);
	//             simd_operation_sum_all(&value3, a->values + i + 12, a->values + i + 14);

	//             i += 16;
	//         } else if (a->size - i >= 12) {
	//             simd_operation_sum_all(&value0, a->values + i, a->values + i + 2);
	//             simd_operation_sum_all(&value1, a->values + i + 4, a->values + i + 6);
	//             simd_operation_sum_all(&value2, a->values + i + 8, a->values + i + 10);

	//             i += 12;
	//         } else if (a->size - i >= 8) {
	//             simd_operation_sum_all(&value0, a->values + i, a->values + i + 2);
	//             simd_operation_sum_all(&value1, a->values + i + 4, a->values + i + 6);

	//             i += 8;
	//         } else if (a->size - i >= 4) {
	//             simd_operation_sum_all(&value0, a->values + i, a->values + i + 2);

	//             i += 4;
	//         } else {
	//             value0 += a->values[i] + a->values[i + 1] + a->values[i + 2];

	//             i += 3;
	//         }
	//     }
	//     values[0] = value0 + value1 + value2 + value3;
	// #else
	double* values  = malloc(size * sizeof(double));

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
	double* ones = malloc(a->size * sizeof(double));

	for (int i = 0; i < a->size; i++) {
		ones[i] = 1;
	}

	gsl_vector_view av = gsl_vector_view_array(a->values, a->size);
	gsl_vector_view b  = gsl_vector_view_array(ones, a->size);
	gsl_blas_ddot(&av.vector, &b.vector, values);

	free(ones);
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
	double one = 1;
	values[0]  = cblas_ddot(a->size, a->values, 1, &one, 0);
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
	double sum = 0;
	for (int i = 0; i < a->size; i++) {
		sum += a->values[i];
	}

	values[0] = sum;
#endif
	// #endif

	tensor->size    = size;
	tensor->ndim    = ndim;
	tensor->shape   = shape;
	tensor->_stride = stride;
	tensor->values  = values;

	return tensor;
}
