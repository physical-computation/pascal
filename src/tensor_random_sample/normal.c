#include <math.h>
#include <stdlib.h>

#include "pascal.h"

double pascal_tensor_random_sample_normal(double mean, double stddev) {
	double z_1 = (double)rand() / RAND_MAX;
	double z_2 = (double)rand() / RAND_MAX;

	double s   = sqrt(-2 * log(z_1)) * cos(2 * M_PI * z_2);

	return (stddev * s) + mean;
}
