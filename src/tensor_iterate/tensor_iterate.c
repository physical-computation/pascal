
#include "pascal.h"

void pascal_tensor_iterate(TensorIterator iterator, Tensor a) {
	for (int i = a->ndim - 1; i >= 0; i--) {
		if (iterator->indexes[i] < a->shape[i] - 1) {
			iterator->indexes[i]++;
			iterator->offset += a->_stride[i];
			break;
		} else {
			iterator->indexes[i] = 0;
			iterator->offset -= a->_stride[i] * (a->shape[i] - 1);
		}
	}
}
