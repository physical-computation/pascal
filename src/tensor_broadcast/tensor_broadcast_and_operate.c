#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_broadcast_and_operate(Tensor a, Tensor b, double (*operation)(double a, double b)) {
	BroadcastOutput b_output = pascal_tensor_broadcast(a, b);
	Tensor          c        = b_output->tensor;

	double*  values          = malloc(sizeof(double) * c->size);
	index_t* indexes         = malloc(sizeof(index_t) * c->ndim);
	for (int i = 0; i < c->ndim; i++) {
		indexes[i] = 0;
	}

	for (int i = 0; i < c->size; i++) {
		index_t a_index = pascal_tensor_linear_index_from_index(indexes, b_output->a_stride, c->ndim);
		index_t b_index = pascal_tensor_linear_index_from_index(indexes, b_output->b_stride, c->ndim);

		values[i]       = operation(a->values[a_index], b->values[b_index]);

		pascal_tensor_iterate_indexes_next(indexes, c->shape, c->ndim);
	}
	free(indexes);

	pascal_tensor_broadcast_output_free(b_output);
	c->values = values;
	return c;
}
