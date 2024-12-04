#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "pascal.h"
#include <sys/time.h>

int main() {
	srand(time(NULL));

	index_t N      = 100;
	index_t dim    = 3;
	index_t ndim   = 10;

	index_t* shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		shape[i] = dim;
	}

	struct timeval stop, start;
	long int       completion_time = 0;
	for (int i = 0; i < N; i++) {
		Tensor a = pascal_tensor_random_uniform(-1, 1, shape, ndim);
		Tensor b = pascal_tensor_random_uniform(-1, 1, (index_t[]){1}, 1);

		gettimeofday(&start, NULL);
		Tensor c = pascal_tensor_multiply(a, b);
		gettimeofday(&stop, NULL);

		completion_time += ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec));

		pascal_tensor_free(a);
		pascal_tensor_free(b);
		pascal_tensor_free(c);
	}
	free(shape);
	completion_time = completion_time / N;
	printf("pascal\t%ld\xC2\xB5s\tmultiply_broadcast\n", completion_time);
}
