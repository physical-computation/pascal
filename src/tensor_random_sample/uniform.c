#include <stdlib.h>

#include "pascal.h"

double pascal_tensor_random_sample_uniform(double min, double max) {
	double s = (double)rand() / RAND_MAX;

	return (s * (max - min)) + min;
}
