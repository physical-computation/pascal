#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

void pascal_tensor_print_values(double array[], index_t size) {
	printf("(");
	for (int i = 0; i < size - 1; i++) {
		printf("%f, ", array[i]);
	}
	printf("%f", array[size - 1]);
	printf("), %u\n", size);
}
