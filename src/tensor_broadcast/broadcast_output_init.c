#include "pascal.h"

BroadcastOutput pascal_tensor_broadcast_output_init() {
	BroadcastOutput b_output = malloc(sizeof(BroadcastOutput_D));
	b_output->tensor         = NULL;
	b_output->a_stride       = NULL;
	b_output->b_stride       = NULL;

	return b_output;
}
