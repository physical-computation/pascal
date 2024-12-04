#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

TensorIterator pascal_tensor_iterator_new(Tensor a) {
	TensorIterator iterator = malloc(sizeof(TensorIterator_D));
	iterator->indexes       = malloc(sizeof(index_t) * a->ndim);
	for (int i = 0; i < a->ndim; i++) {
		iterator->indexes[i] = 0;
	}

	iterator->offset = 0;

	return iterator;
}
