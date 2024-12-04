#include "arbiter.h"

#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static void test_less_num() {
	double* x = NULL;
	double* y = NULL;

	int num_data_points =
			load_data("tests/load_data/data_less_num.dat", &x, &y);
	arbiter_assert(num_data_points == 1);

	free(x);
	free(y);
}

static void test_more_num() {
	double* x = NULL;
	double* y = NULL;

	int num_data_points =
			load_data("tests/load_data/data_more_num.dat", &x, &y);

	arbiter_assert(num_data_points == 2);

	free(x);
	free(y);
}

static void test_load() {
	double* x           = NULL;
	double* y           = NULL;

	int num_data_points = load_data("tests/load_data/data.dat", &x, &y);

	arbiter_assert(num_data_points == 2);

	arbiter_assert(x[0] == 1.0);
	arbiter_assert((int)(x[1] * 10) == 13);

	arbiter_assert((int)(y[0] * 10) == 20);
	arbiter_assert((int)(y[1] * 10) == 23);

	free(x);
	free(y);
}

#define NUM_TESTS 3

int main() {
	void (*tests[NUM_TESTS])() = {test_load, test_less_num, test_more_num};

	arbiter_run_tests(NUM_TESTS, "load_data", tests);
}
