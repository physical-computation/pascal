#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "pascal.h"
#include <sys/time.h>

int main() {
	srand(time(NULL));

	index_t N        = 1000;

	index_t ndim     = 2;
	index_t dim      = 100000;
	index_t shape[2] = {dim, 1};

	Tensor a         = pascal_tensor_random_uniform(-1, 1, shape, ndim);
	Tensor b         = pascal_tensor_random_uniform(-1, 1, shape, ndim);

	Tensor         c;
	struct timeval stop, start;
	long int       completion_time = 0;
	for (int i = 0; i < N; i++) {
		gettimeofday(&start, NULL);
		c = pascal_tensor_dot(a, b);
		gettimeofday(&stop, NULL);

		completion_time += ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec));
	}

	completion_time = completion_time / N;
	printf("pascal\t%ld\xC2\xB5s\tdot\n", completion_time);

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}
