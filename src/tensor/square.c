#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_square(Tensor a) {
	if (a->_transpose_map != NULL || a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	Tensor b       = pascal_tensor_init();

	b->ndim        = a->ndim;
	b->size        = a->size;
	index_t* shape = malloc((a->ndim + 1) * sizeof(index_t));
	for (int i = 0; i < a->ndim; i++) {
		shape[i] = a->shape[i];
	}
	b->shape       = shape;
	b->_stride     = pascal_tensor_utils_default_stride(shape, a->ndim);

	double* values = malloc(a->size * sizeof(double));

	// #if TENSOR_USE_SIMD
	//     int i = 0;

	//     bool excess = a->size % 2 == 1;

	//     for (int i = 0; i < a->size / 2; i++) {
	//         int _i = i * 2;

	//         float64x2_t v;
	//         float64x2_t _a = vld1q_f64(a->values + _i);
	//         v = vmulq_f64(_a, _a);

	//         vst1q_f64(values + _i, v);
	//     }

	//     if (excess) {
	//         int i = a->size - 1;
	//         values[i] = a->values[i] * a->values[i];
	//     }
	// #else
	for (int i = 0; i < a->size; i++) {
		values[i] = a->values[i] * a->values[i];
	}
	// #endif

	b->values                 = values;

	b->_transpose_map         = a->_transpose_map;
	b->_transpose_map_inverse = a->_transpose_map_inverse;
	return b;
}
