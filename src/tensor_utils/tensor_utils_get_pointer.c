#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

double* pascal_tensor_utils_get_pointer(Tensor a, index_t indexes[]) {
	index_t linear_index = 0;
	//  TODO: Is there a way to that the number of args is correct?
	for (int i = 0; i < a->ndim; i++) {
		index_t index = indexes[i];
		pascal_tensor_assert(index < a->shape[i], "Index is larger than ndim. Please check");

		linear_index += index * a->_stride[i];
	}

	return &(a->values[linear_index]);
}
