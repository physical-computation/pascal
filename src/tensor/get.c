#include "pascal.h"

double pascal_tensor_get(Tensor a, index_t indexes[]) {
	index_t linear_index = pascal_tensor_linear_index_from_index(indexes, a->_stride, a->ndim);

	return a->values[linear_index];
}
