#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_linalg_cholesky(Tensor a) {
	pascal_tensor_assert(a->shape[a->ndim - 1] == a->shape[a->ndim - 1], "The last 2 dimensions must be equal (square matrix)\n");

	if (a->_transpose_map != NULL) {
		pascal_tensor_utils_unravel_and_replace(a);
	}

	index_t N               = a->shape[a->ndim - 1];
	index_t fixed_axis_size = N * N;

	double*  values         = calloc(a->size, sizeof(double));
	index_t* indexes        = calloc(N - 2, sizeof(index_t));

	Tensor   c              = pascal_tensor_init();
	index_t* shape          = malloc((a->ndim + 1) * sizeof(index_t));
	for (int i = 0; i < a->ndim; i++) {
		shape[i] = a->shape[i];
	}
	index_t* stride = pascal_tensor_utils_default_stride(shape, a->ndim);

	for (int k = 0; k < a->size / fixed_axis_size; k++) {
		index_t values_offset = k * fixed_axis_size;

		double* _a            = a->values + values_offset;
		double* _values       = values + values_offset;

		for (int i = 0; i < a->ndim; i++) {
			for (int j = 0; j < i + 1; j++) {

				index_t offset = i * N + j;

				double sum     = _a[offset];

				int l;
				for (l = 0; l < j; l++) {
					index_t offset1 = i * N + l; // Assuming row major
					index_t offset2 = j * N + l; // Assuming row major

					sum -= (_values[offset1] * _values[offset2]);
				}

				if (i == j) {
					_values[offset] = sqrt(sum);
					continue;
				}

				index_t offset3 = j * N + l;
				_values[offset] = sum / _values[offset3];
			}
		}

		pascal_tensor_iterate_indexes_next(indexes, shape, a->ndim - 2);
	}

	free(indexes);

	c->ndim    = a->ndim;
	c->size    = a->size;
	c->shape   = shape;
	c->_stride = stride;
	c->values  = values;

	return c;
}
