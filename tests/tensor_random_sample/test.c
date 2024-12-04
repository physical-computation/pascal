#include "arbiter.h"

#include "pascal.h"

#define NUM_TESTS 2

static void test_pascal_tensor_random_sample_uniform() {
	double min    = 0;
	double max    = 1;

	double sample = pascal_tensor_random_sample_uniform(min, max);

	arbiter_assert(sample >= min);
	arbiter_assert(sample <= max);
}

static void test_pascal_tensor_random_sample_normal() {
	double mean   = 0;
	double stddev = 1;

	double sample = pascal_tensor_random_sample_normal(mean, stddev);
}

int main() {
	void (*tests[NUM_TESTS])() = {
			test_pascal_tensor_random_sample_uniform,
			test_pascal_tensor_random_sample_normal};

	arbiter_run_tests(NUM_TESTS, "pascal_tensor_random_sample", tests);
}
