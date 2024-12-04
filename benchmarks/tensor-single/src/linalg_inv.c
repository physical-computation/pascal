#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "pascal.h"
#include <sys/time.h>

int main() {
	srand(time(NULL));

	index_t N        = 1000;

	index_t ndim     = 2;
	index_t dim      = 50;
	index_t shape[2] = {dim, dim};

	Tensor a         = pascal_tensor_random_uniform(-1, 1, shape, ndim);

	Tensor         b;
	struct timeval stop, start;
	long int       completion_time = 0;
	for (int i = 0; i < N; i++) {
		gettimeofday(&start, NULL);
		b = pascal_tensor_linalg_inv(a);
		gettimeofday(&stop, NULL);

		completion_time += ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec));
	}

	completion_time = completion_time / N;
	printf("pascal\t%ld\xC2\xB5s\tlinalg_inv\n", completion_time);

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}
