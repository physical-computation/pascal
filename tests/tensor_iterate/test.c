#include "arbiter.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static void test_pascal_tensor_iterator_new() {
	index_t ndim            = 3;
	index_t shape[3]        = {2, 3, 4};

	Tensor         a        = pascal_tensor_random_uniform(0, 1, shape, ndim);
	TensorIterator iterator = pascal_tensor_iterator_new(a);

	arbiter_assert(iterator->offset == 0);
	for (int i = 0; i < ndim; i++) {
		arbiter_assert(iterator->indexes[i] == 0);
	}

	pascal_tensor_free(a);
	pascal_tensor_iterator_free(iterator);
}

static void test_pascal_tensor_iterator_copy() {
	index_t  ndim           = 3;
	index_t* indexes        = malloc(ndim * sizeof(index_t));
	indexes[0]              = 2;
	indexes[1]              = 3;
	indexes[2]              = 4;

	index_t offset          = 0;

	TensorIterator iterator = malloc(sizeof(TensorIterator_D));
	iterator->offset        = ndim;
	iterator->indexes       = indexes;

	TensorIterator copy     = pascal_tensor_iterator_copy(iterator, ndim);

	arbiter_assert(copy->offset == iterator->offset);
	for (int i = 0; i < ndim; i++) {
		arbiter_assert(copy->indexes[i] == iterator->indexes[i]);
	}

	pascal_tensor_iterator_free(iterator);
	pascal_tensor_iterator_free(copy);
}

static void test_pascal_tensor_iterate_indexes_next() {
	index_t ndim     = 5;
	index_t shape[5] = {2, 3, 4, 3, 2};
	index_t size     = 144;

	index_t* indexes = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		indexes[i] = 0;
	}

	index_t expected_indexes[144][5] = {
			{0, 0, 0, 0, 0},
			{0, 0, 0, 0, 1},
			{0, 0, 0, 1, 0},
			{0, 0, 0, 1, 1},
			{0, 0, 0, 2, 0},
			{0, 0, 0, 2, 1},
			{0, 0, 1, 0, 0},
			{0, 0, 1, 0, 1},
			{0, 0, 1, 1, 0},
			{0, 0, 1, 1, 1},
			{0, 0, 1, 2, 0},
			{0, 0, 1, 2, 1},
			{0, 0, 2, 0, 0},
			{0, 0, 2, 0, 1},
			{0, 0, 2, 1, 0},
			{0, 0, 2, 1, 1},
			{0, 0, 2, 2, 0},
			{0, 0, 2, 2, 1},
			{0, 0, 3, 0, 0},
			{0, 0, 3, 0, 1},
			{0, 0, 3, 1, 0},
			{0, 0, 3, 1, 1},
			{0, 0, 3, 2, 0},
			{0, 0, 3, 2, 1},
			{0, 1, 0, 0, 0},
			{0, 1, 0, 0, 1},
			{0, 1, 0, 1, 0},
			{0, 1, 0, 1, 1},
			{0, 1, 0, 2, 0},
			{0, 1, 0, 2, 1},
			{0, 1, 1, 0, 0},
			{0, 1, 1, 0, 1},
			{0, 1, 1, 1, 0},
			{0, 1, 1, 1, 1},
			{0, 1, 1, 2, 0},
			{0, 1, 1, 2, 1},
			{0, 1, 2, 0, 0},
			{0, 1, 2, 0, 1},
			{0, 1, 2, 1, 0},
			{0, 1, 2, 1, 1},
			{0, 1, 2, 2, 0},
			{0, 1, 2, 2, 1},
			{0, 1, 3, 0, 0},
			{0, 1, 3, 0, 1},
			{0, 1, 3, 1, 0},
			{0, 1, 3, 1, 1},
			{0, 1, 3, 2, 0},
			{0, 1, 3, 2, 1},
			{0, 2, 0, 0, 0},
			{0, 2, 0, 0, 1},
			{0, 2, 0, 1, 0},
			{0, 2, 0, 1, 1},
			{0, 2, 0, 2, 0},
			{0, 2, 0, 2, 1},
			{0, 2, 1, 0, 0},
			{0, 2, 1, 0, 1},
			{0, 2, 1, 1, 0},
			{0, 2, 1, 1, 1},
			{0, 2, 1, 2, 0},
			{0, 2, 1, 2, 1},
			{0, 2, 2, 0, 0},
			{0, 2, 2, 0, 1},
			{0, 2, 2, 1, 0},
			{0, 2, 2, 1, 1},
			{0, 2, 2, 2, 0},
			{0, 2, 2, 2, 1},
			{0, 2, 3, 0, 0},
			{0, 2, 3, 0, 1},
			{0, 2, 3, 1, 0},
			{0, 2, 3, 1, 1},
			{0, 2, 3, 2, 0},
			{0, 2, 3, 2, 1},
			{1, 0, 0, 0, 0},
			{1, 0, 0, 0, 1},
			{1, 0, 0, 1, 0},
			{1, 0, 0, 1, 1},
			{1, 0, 0, 2, 0},
			{1, 0, 0, 2, 1},
			{1, 0, 1, 0, 0},
			{1, 0, 1, 0, 1},
			{1, 0, 1, 1, 0},
			{1, 0, 1, 1, 1},
			{1, 0, 1, 2, 0},
			{1, 0, 1, 2, 1},
			{1, 0, 2, 0, 0},
			{1, 0, 2, 0, 1},
			{1, 0, 2, 1, 0},
			{1, 0, 2, 1, 1},
			{1, 0, 2, 2, 0},
			{1, 0, 2, 2, 1},
			{1, 0, 3, 0, 0},
			{1, 0, 3, 0, 1},
			{1, 0, 3, 1, 0},
			{1, 0, 3, 1, 1},
			{1, 0, 3, 2, 0},
			{1, 0, 3, 2, 1},
			{1, 1, 0, 0, 0},
			{1, 1, 0, 0, 1},
			{1, 1, 0, 1, 0},
			{1, 1, 0, 1, 1},
			{1, 1, 0, 2, 0},
			{1, 1, 0, 2, 1},
			{1, 1, 1, 0, 0},
			{1, 1, 1, 0, 1},
			{1, 1, 1, 1, 0},
			{1, 1, 1, 1, 1},
			{1, 1, 1, 2, 0},
			{1, 1, 1, 2, 1},
			{1, 1, 2, 0, 0},
			{1, 1, 2, 0, 1},
			{1, 1, 2, 1, 0},
			{1, 1, 2, 1, 1},
			{1, 1, 2, 2, 0},
			{1, 1, 2, 2, 1},
			{1, 1, 3, 0, 0},
			{1, 1, 3, 0, 1},
			{1, 1, 3, 1, 0},
			{1, 1, 3, 1, 1},
			{1, 1, 3, 2, 0},
			{1, 1, 3, 2, 1},
			{1, 2, 0, 0, 0},
			{1, 2, 0, 0, 1},
			{1, 2, 0, 1, 0},
			{1, 2, 0, 1, 1},
			{1, 2, 0, 2, 0},
			{1, 2, 0, 2, 1},
			{1, 2, 1, 0, 0},
			{1, 2, 1, 0, 1},
			{1, 2, 1, 1, 0},
			{1, 2, 1, 1, 1},
			{1, 2, 1, 2, 0},
			{1, 2, 1, 2, 1},
			{1, 2, 2, 0, 0},
			{1, 2, 2, 0, 1},
			{1, 2, 2, 1, 0},
			{1, 2, 2, 1, 1},
			{1, 2, 2, 2, 0},
			{1, 2, 2, 2, 1},
			{1, 2, 3, 0, 0},
			{1, 2, 3, 0, 1},
			{1, 2, 3, 1, 0},
			{1, 2, 3, 1, 1},
			{1, 2, 3, 2, 0},
			{1, 2, 3, 2, 1}};

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < ndim; j++) {
			arbiter_assert(indexes[j] == expected_indexes[i][j]);
		}
		pascal_tensor_iterate_indexes_next(indexes, shape, ndim);
	}

	free(indexes);
}

static void test_pascal_tensor_iterate_current() {
	index_t ndim            = 4;
	index_t shape[4]        = {1, 2, 3, 4};
	double  values[24]      = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor         a        = pascal_tensor_new(values, shape, ndim);
	TensorIterator iterator = pascal_tensor_iterator_new(a);

	for (int i = 0; i < a->size * 2; i++) {
		arbiter_assert(values[i % a->size] == pascal_tensor_iterate_current(iterator, a));
		pascal_tensor_iterate(iterator, a);
	}

	pascal_tensor_free(a);
	pascal_tensor_iterator_free(iterator);
}

static void test_pascal_tensor_iterate() {
	index_t ndim                    = 4;
	index_t shape[4]                = {1, 2, 3, 4};
	double  values[24]              = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor         a                = pascal_tensor_new(values, shape, ndim);
	TensorIterator iterator         = pascal_tensor_iterator_new(a);

	index_t expected_indexes[24][4] = {
			{0, 0, 0, 0},
			{0, 0, 0, 1},
			{0, 0, 0, 2},
			{0, 0, 0, 3},
			{0, 0, 1, 0},
			{0, 0, 1, 1},
			{0, 0, 1, 2},
			{0, 0, 1, 3},
			{0, 0, 2, 0},
			{0, 0, 2, 1},
			{0, 0, 2, 2},
			{0, 0, 2, 3},
			{0, 1, 0, 0},
			{0, 1, 0, 1},
			{0, 1, 0, 2},
			{0, 1, 0, 3},
			{0, 1, 1, 0},
			{0, 1, 1, 1},
			{0, 1, 1, 2},
			{0, 1, 1, 3},
			{0, 1, 2, 0},
			{0, 1, 2, 1},
			{0, 1, 2, 2},
			{0, 1, 2, 3},
	};

	for (int i = 0; i < a->size * 2; i++) {
		for (int j = 0; j < ndim; j++) {
			arbiter_assert(iterator->indexes[j] == expected_indexes[i % a->size][j]);
		}

		arbiter_assert(values[i % a->size] == pascal_tensor_iterate_current(iterator, a));
		pascal_tensor_iterate(iterator, a);
	}

	pascal_tensor_free(a);
	pascal_tensor_iterator_free(iterator);
}

static void test_pascal_tensor_iterate_next() {
	index_t ndim            = 4;
	index_t shape[4]        = {1, 2, 3, 4};
	double  values[24]      = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor         a        = pascal_tensor_new(values, shape, ndim);
	TensorIterator iterator = pascal_tensor_iterator_new(a);

	for (int i = 1; i < (a->size * 2) - 1; i++) {
		arbiter_assert(values[i % a->size] == pascal_tensor_iterate_next(iterator, a));
	}

	pascal_tensor_free(a);
	pascal_tensor_iterator_free(iterator);
}

#define NUM_TESTS 6

int main() {
	void (*tests[NUM_TESTS])() = {
			test_pascal_tensor_iterator_copy,
			test_pascal_tensor_iterator_new,
			test_pascal_tensor_iterate_indexes_next,
			test_pascal_tensor_iterate_current,
			test_pascal_tensor_iterate,
			test_pascal_tensor_iterate_next,
	};
	arbiter_run_tests(NUM_TESTS, "pascal_tensor_iterate", tests);
}
