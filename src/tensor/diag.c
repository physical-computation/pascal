#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

// This broadcasts across every dimension except the last two
Tensor pascal_tensor_diag(Tensor a) {
	pascal_tensor_assert(a->ndim >= 2, "Tensor must be at least 2 dimensional\n");
	pascal_tensor_assert(a->shape[a->ndim - 1] == a->shape[a->ndim - 2], "Matrix in last 2 dims must be square\n");

	Tensor tensor   = pascal_tensor_init();

	index_t  _ndim  = a->ndim - 1;
	index_t* _shape = malloc((_ndim) * sizeof(index_t));
	for (int i = 0; i < _ndim; i++) {
		_shape[i] = a->shape[i];
	}

	index_t  _size     = pascal_tensor_utils_size_from_shape(_shape, _ndim);
	index_t* _stride   = pascal_tensor_utils_default_stride(_shape, _ndim);

	double*  _values   = malloc(_size * sizeof(double));
	index_t* a_indexes = malloc(a->ndim * sizeof(index_t));
	index_t* indexes   = malloc(_ndim * sizeof(index_t));
	for (int i = 0; i < _ndim; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < _size; i++) {
		for (int j = 0; j < _ndim; j++) {
			a_indexes[j] = indexes[j];
		}
		a_indexes[a->ndim - 1] = indexes[_ndim - 1];

		_values[i]             = pascal_tensor_get(a, a_indexes);

		pascal_tensor_iterate_indexes_next(indexes, _shape, _ndim);
	}
	free(a_indexes);
	free(indexes);

	tensor->ndim    = _ndim;
	tensor->shape   = _shape;
	tensor->_stride = _stride;
	tensor->size    = _size;
	tensor->values  = _values;

	return tensor;
}
