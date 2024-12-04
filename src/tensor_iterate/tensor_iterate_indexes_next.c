#include "pascal.h"

void pascal_tensor_iterate_indexes_next(index_t* indexes, index_t* shape, index_t ndim) {
	for (int i = ndim - 1; i >= 0; i--) {
		if (indexes[i] < shape[i] - 1) {
			indexes[i]++;
			break;
		} else {
			indexes[i] = 0;
		}
	}
}
