#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

/*
Currently supports only DxD tensors.
*/
Tensor pascal_tensor_eye(index_t n) {
	double* _values = malloc(sizeof(double) * n * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				_values[i * n + j] = 1.0;
			} else {
				_values[i * n + j] = 0.0;
			}
		}
	}

	Tensor tensor = pascal_tensor_new_no_malloc(_values, (index_t[]){n, n}, 2);
	return tensor;
}
