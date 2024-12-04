#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static double operation_reciprocal(double a) {
	return 1 / a;
}

Tensor pascal_tensor_reciprocal(Tensor a) {
	Tensor  tensor  = pascal_tensor_init();
	index_t ndim    = a->ndim;

	index_t  size   = a->size;
	index_t* shape  = malloc(ndim * sizeof(index_t));
	index_t* stride = malloc(ndim * sizeof(index_t));
	for (int i = 0; i < ndim; i++) {
		shape[i]  = a->shape[i];
		stride[i] = a->_stride[i];
	}

	double* values = malloc(size * sizeof(double));

	// #if TENSOR_USE_SIMD
	//     float64x2_t one = vdupq_n_f64(1);
	//     for (int i = 0; i < a->size; i += 2) {
	//         float64x2_t _a = vld1q_f64(a->values + i);
	//         float64x2_t v = vdivq_f64(one, _a);
	//         vst1q_f64(values + i, v);
	//     }
	// #else
	for (int i = 0; i < a->size; i++) {
		values[i] = operation_reciprocal(a->values[i]);
	}
	// #endif

	tensor->size                   = size;
	tensor->ndim                   = ndim;
	tensor->shape                  = shape;
	tensor->_stride                = stride;
	tensor->_transpose_map         = a->_transpose_map;
	tensor->_transpose_map_inverse = a->_transpose_map_inverse;
	tensor->values                 = values;

	return tensor;
}
