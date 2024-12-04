#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

void pascal_tensor_iterator_free(TensorIterator iterator) {
	if (iterator == NULL) {
		return;
	}

	free(iterator->indexes);
	free(iterator);
}
