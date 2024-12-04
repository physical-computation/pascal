#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_init() {
	Tensor tensor                  = malloc(sizeof(Tensor_D));

	tensor->size                   = 0;
	tensor->ndim                   = 0;
	tensor->shape                  = NULL;
	tensor->_stride                = NULL;
	tensor->_transpose_map         = NULL;
	tensor->_transpose_map_inverse = NULL;
	tensor->values                 = NULL;

	return tensor;
}
