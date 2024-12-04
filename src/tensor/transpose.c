#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_transpose(Tensor a, index_t transpose_map[]) {
	index_t* shape                  = malloc(a->ndim * sizeof(index_t));
	index_t* stride                 = malloc(a->ndim * sizeof(index_t));

	index_t* _transpose_map         = malloc(a->ndim * sizeof(index_t));
	index_t* _transpose_map_inverse = malloc(a->ndim * sizeof(index_t));

	if (a->_transpose_map == NULL) {
		for (int i = 0; i < a->ndim; i++) {
			shape[i]                                 = a->shape[transpose_map[i]];
			stride[i]                                = a->_stride[transpose_map[i]];

			_transpose_map[i]                        = transpose_map[i];
			_transpose_map_inverse[transpose_map[i]] = i;
		}
	} else {
		for (int i = 0; i < a->ndim; i++) {
			shape[i]          = a->shape[transpose_map[i]];
			stride[i]         = a->_stride[transpose_map[i]];
			_transpose_map[i] = a->_transpose_map[transpose_map[i]];
		}
		// TODO: Can this be moved to the loop just above, rather than on a separate loop?
		for (int i = 0; i < a->ndim; i++) {
			_transpose_map_inverse[_transpose_map[i]] = i;
		}
	}

	Tensor tensor = pascal_tensor_new(a->values, shape, a->ndim);
	free(tensor->_stride);
	free(shape);

	tensor->_stride                = stride;
	tensor->_transpose_map         = _transpose_map;
	tensor->_transpose_map_inverse = _transpose_map_inverse;

	// TODO: This is a bit of a hack. It seems that transpose is a pain, and I will end up having to unravel later on. So I'm just unraveling here for now.
	// if (tensor->_transpose_map != NULL || tensor->_transpose_map != NULL) {
	//     pascal_tensor_utils_unravel_and_replace(tensor);
	// }

	return tensor;
}
