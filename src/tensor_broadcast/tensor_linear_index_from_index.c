#include "pascal.h"

#if TENSOR_USE_SIMD
	#include "arm_neon.h"
#endif

index_t pascal_tensor_linear_index_from_index(index_t index[], index_t stride[], index_t ndim) {
#if TENSOR_USE_SIMD
	int     i            = 0;
	index_t linear_index = 0;
	while (i < ndim) {
		if (ndim - i >= 4) {
			uint32x4_t v;

			uint32x4_t s   = vld1q_u32(stride + i);
			uint32x4_t idx = vld1q_u32(index + i);

			v              = vmulq_u32(idx, s);
			linear_index += vaddvq_u32(v);

			i += 4;
		} else {
			for (int j = 0; j < 4; j++) {
				linear_index += index[i + j] * stride[i + j];
			}

			i += 3;
		}
	}
#else
	index_t linear_index = 0;
	for (int i = 0; i < ndim; i++) {
		linear_index += index[i] * stride[i];
	}
#endif
	return linear_index;
}
