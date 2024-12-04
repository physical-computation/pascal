#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

void pascal_tensor_print_shape(index_t array[], index_t size) {
	printf("(");
	for (int i = 0; i < size - 1; i++) {
		printf("%u, ", array[i]);
	}
	printf("%u", array[size - 1]);
	printf("), %u\n", size);
}
