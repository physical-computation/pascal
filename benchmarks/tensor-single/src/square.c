#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "pascal.h"
#include <sys/time.h>

int main() {
	srand(time(NULL));

	index_t N        = 10000;

	index_t ndim     = 2;
	index_t dim      = 200;
	index_t shape[2] = {dim, dim};

	struct timeval stop, start;
	long int       completion_time = 0;
	for (int i = 0; i < N; i++) {
		Tensor a = pascal_tensor_random_uniform(-1, 1, shape, ndim);

		Tensor b;

		gettimeofday(&start, NULL);
		b = pascal_tensor_square(a);
		gettimeofday(&stop, NULL);

		completion_time += ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec));

		pascal_tensor_free(a);
		pascal_tensor_free(b);
	}

	completion_time = completion_time / N;
	printf("pascal\t%ld\xC2\xB5s\tsquare\n", completion_time);
}
