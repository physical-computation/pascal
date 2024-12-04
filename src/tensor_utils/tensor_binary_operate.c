#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_binary_operate(Tensor a, Tensor b, double (*operation)(double, double)) {
	if (a->_transpose_map != NULL || a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	if (b->_transpose_map != NULL || b->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(b);
	}

	Tensor c       = pascal_tensor_init();

	c->ndim        = a->ndim;
	c->size        = a->size;
	index_t* shape = malloc((a->ndim + 1) * sizeof(index_t));
	for (int i = 0; i < a->ndim; i++) {
		shape[i] = a->shape[i];
	}
	c->shape       = shape;
	c->_stride     = pascal_tensor_utils_default_stride(shape, a->ndim);

	// double* _a = a->values;
	// double* _b = b->values;

	double* values = malloc(a->size * sizeof(double));
	int     i      = 0;

	// bool excess = a->size % 2 == 1;

	// While ugly, this way of doing it seems faster...
	// for (i = 0; i < a->size / 2; i++) {
	//     int _i = i * 2;
	//     operation(values + _i, _a + _i, _b + _i, false);
	// }

	// if (excess) {
	//     int i = a->size - 1;
	//     operation(values + i, _a + i, _b + i, excess);
	// }

	for (i = 0; i < a->size; i++) {
		values[i] = operation(a->values[i], b->values[i]);
	}

	c->values = values;

	return c;
}
