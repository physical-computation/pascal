#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"
TensorIterator pascal_tensor_iterator_copy(TensorIterator iterator, index_t ndim) {
	TensorIterator copy = malloc(sizeof(TensorIterator_D));
	copy->indexes       = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		copy->indexes[i] = iterator->indexes[i];
	}

	copy->offset = iterator->offset;

	return copy;
}
