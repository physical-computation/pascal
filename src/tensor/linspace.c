#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

/*
Currently supports only Dx1 tensors.
*/
Tensor pascal_tensor_linspace(double start, double end, index_t num) {
	double* _values = malloc(sizeof(double) * num);
	for (int i = 0; i < num; i++) {
		_values[i] = start + (end - start) * i / (num - 1);
	}

	Tensor tensor = pascal_tensor_new_no_malloc(_values, (index_t[]){num, 1}, 2);

	return tensor;
}
