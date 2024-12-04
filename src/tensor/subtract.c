#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_USE_SIMD
	#include <arm_neon.h>
#endif

static double operation_subtract_scalar(double a, double b) {
	return a - b;
}

// static void operation_subtract(double* out, double* a, double* b, bool excess) {
// 	if (excess) {
// 		out[0] = a[0] - b[0];
// 		return;
// 	}

// #if TENSOR_USE_SIMD
// 	float64x2_t v;

// 	float64x2_t _a = vld1q_f64(a);
// 	float64x2_t _b = vld1q_f64(b);

// 	v              = vsubq_f64(_a, _b);

// 	vst1q_f64(out, v);
// #else
// 	out[0] = a[0] - b[0];
// 	out[1] = a[1] - b[1];
// #endif
// }

Tensor pascal_tensor_subtract(Tensor a, Tensor b) {
	if (pascal_tensor_broadcast_is_needed(a, b)) {
		return pascal_tensor_broadcast_and_operate(a, b, operation_subtract_scalar);
	}

	return pascal_tensor_binary_operate(a, b, operation_subtract_scalar);
}
