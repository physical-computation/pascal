#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static void pascal_tensor_print_recursion(Tensor a, index_t indexes[], index_t dim, bool final) {
	if (dim == a->ndim - 1) {
		printf("[");
	} else {
		printf("[");
	}
	for (int i = 0; i < a->shape[dim]; i++) {
		indexes[dim] = i;
		if (dim == a->ndim - 1) {
#if TENSOR_PRINT_VERBOSE
			if (i == a->shape[dim] - 1) {
				printf("%.*e", DECIMAL_DIG, pascal_tensor_get(a, indexes));
			} else {
				printf("%.*e, ", DECIMAL_DIG, pascal_tensor_get(a, indexes));
			}
#else
			if (i == a->shape[dim] - 1) {
				printf("%.12lf", pascal_tensor_get(a, indexes));
			} else {
				printf("%.12lf, ", pascal_tensor_get(a, indexes));
			}
#endif
		} else {
			if (i == a->shape[dim] - 1) {
				pascal_tensor_print_recursion(a, indexes, dim + 1, true);
			} else {
				pascal_tensor_print_recursion(a, indexes, dim + 1, false);
			}
		}
	}
	if (dim == a->ndim - 1) {
		if (final) {
			printf("]");
		} else {
			printf("],\n%*s", (int)dim, "");
		}
	} else {
		if (final) {
			printf("]");
		} else {
			printf("],\n\n%*s", (int)dim, "");
		}
	}
}

static void pascal_tensor_print_one_dim(Tensor a) {
	printf("[");
	for (int i = 0; i < a->shape[0]; i++) {
#if TENSOR_PRINT_VERBOSE
		if (i == a->shape[0] - 1) {
			printf("%.*e\n", DECIMAL_DIG, pascal_tensor_get(a, (index_t[]){i}));
		} else {
			printf("%.*e,\n", DECIMAL_DIG, pascal_tensor_get(a, (index_t[]){i}));
		}
#else
		if (i == a->shape[0] - 1) {
			printf("%.12lf", pascal_tensor_get(a, (index_t[]){i}));
		} else {
			printf("%.12lf, ", pascal_tensor_get(a, (index_t[]){i}));
		}
#endif
	}
	printf("]\n");
}

void pascal_tensor_print(Tensor a) {
	if (a->ndim == 1) {
		pascal_tensor_print_one_dim(a);
		return;
	}

	index_t  dim     = 0;
	index_t* indexes = malloc(sizeof(index_t) * a->ndim);
	printf("[");
	for (int i = 0; i < a->shape[dim]; i++) {
		indexes[dim] = i;
		if (dim == a->ndim - 1) {
			printf("%lf", pascal_tensor_get(a, indexes));
		} else {
			if (i == a->shape[dim] - 1) {
				pascal_tensor_print_recursion(a, indexes, dim + 1, true);
			} else {
				pascal_tensor_print_recursion(a, indexes, dim + 1, false);
			}
		}
	}
	printf("]\n");

	free(indexes);
}
