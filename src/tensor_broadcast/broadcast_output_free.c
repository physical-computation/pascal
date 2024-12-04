#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_USE_SIMD
	#include "arm_neon.h"
#endif

void pascal_tensor_broadcast_output_free(BroadcastOutput b_output) {
	free(b_output->a_stride);
	free(b_output->b_stride);
	free(b_output);
}
