#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

void pascal_tensor_free(Tensor tensor) {
	free(tensor->shape);
	free(tensor->_stride);
	free(tensor->_transpose_map);
	free(tensor->_transpose_map_inverse);
	free(tensor->values);

	free(tensor);
}
