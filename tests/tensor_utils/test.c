#include "arbiter.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pascal.h"

static void test_pascal_tensor_utils_apply_transpose_map() {
}

static void test_get_size_from_shape() {
	index_t shape[4] = {1, 2, 3, 6};
	index_t ndim     = 4;

	arbiter_assert(pascal_tensor_utils_size_from_shape(shape, ndim) == 36);
}

static void test_pascal_tensor_utils_shapes_equal() {
	double  repeated_value = 0.0;
	index_t shape[3]       = {2, 3, 4};

	Tensor a               = pascal_tensor_new_repeat(repeated_value, shape, 3);
	Tensor b               = pascal_tensor_new_repeat(repeated_value, shape, 3);

	index_t shape2[4]      = {2, 3, 4, 5};
	Tensor  c              = pascal_tensor_new_repeat(repeated_value, shape2, 4);

	index_t shape3[3]      = {2, 3, 5};
	Tensor  d              = pascal_tensor_new_repeat(repeated_value, shape3, 3);

	arbiter_assert(pascal_tensor_utils_shapes_equal(a, b));
	arbiter_assert(!pascal_tensor_utils_shapes_equal(a, c));
	arbiter_assert(!pascal_tensor_utils_shapes_equal(a, d));

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
	pascal_tensor_free(d);
}

static void test_get_stride() {
	index_t shape[4]           = {3, 1, 5, 6};
	index_t ndim               = 4;

	index_t* stride            = pascal_tensor_utils_default_stride(shape, ndim);

	index_t expected_stride[4] = {30, 30, 6, 1};
	for (int i = 0; i < ndim; i++) {
		arbiter_assert(stride[i] == expected_stride[i]);
	}

	free(stride);
}

static void test_pascal_tensor_utils_index_from_linear_index_transpose_safe() {
	index_t shape[3]                  = {3, 2, 2};
	index_t ndim                      = 3;

	Tensor a                          = pascal_tensor_ones(shape, ndim);
	Tensor a_t                        = pascal_tensor_transpose(a, (index_t[]){0, 2, 1});

	index_t expected_indexes_a[12][3] = {
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 0},
			{0, 1, 1},
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 0},
			{1, 1, 1},
			{2, 0, 0},
			{2, 0, 1},
			{2, 1, 0},
			{2, 1, 1},
	};

	index_t expected_indexes_a_t[12][3] = {
			{0, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
			{0, 1, 1},
			{1, 0, 0},
			{1, 1, 0},
			{1, 0, 1},
			{1, 1, 1},
			{2, 0, 0},
			{2, 1, 0},
			{2, 0, 1},
			{2, 1, 1},
	};

	for (int i = 0; i < 12; i++) {
		index_t* indexes_a   = pascal_tensor_utils_index_from_linear_index_transpose_safe(i, a);
		index_t* indexes_a_t = pascal_tensor_utils_index_from_linear_index_transpose_safe(i, a_t);
		for (int j = 0; j < ndim; j++) {
			arbiter_assert(indexes_a[j] == expected_indexes_a[i][j]);
			arbiter_assert(indexes_a_t[j] == expected_indexes_a_t[i][j]);
		}

		free(indexes_a);
		free(indexes_a_t);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
}

static void test_pascal_tensor_utils_index_from_linear_index() {
	index_t shape[4]     = {3, 1, 5, 6};
	index_t ndim         = 4;
	index_t linear_index = 87;

	index_t* stride      = pascal_tensor_utils_default_stride(shape, ndim);
	index_t* indexes     = malloc(ndim * sizeof(index_t));
	pascal_tensor_utils_index_from_linear_index(indexes, linear_index, stride, ndim);
	index_t expected_indexes[4] = {2, 0, 4, 3};

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(indexes[i] == expected_indexes[i]);
	}

	free(stride);
	free(indexes);
}

static void test_pascal_tensor_utils_get_convolution_start_index() {
	index_t indexes_1[5]      = {0, 0, 0, 0, 0};
	index_t stride_1[3]       = {0, 2, 2};

	index_t* out_1            = pascal_tensor_utils_get_convolution_start_index(indexes_1, stride_1, 5, 3);

	index_t expected_out_1[5] = {0, 0, 0, 0, 0};
	for (int i = 0; i < 5; i++) {
		arbiter_assert(out_1[i] == expected_out_1[i]);
	}

	index_t indexes_2[5]      = {0, 0, 2, 0, 0};
	index_t stride_2[3]       = {0, 2, 2};

	index_t* out_2            = pascal_tensor_utils_get_convolution_start_index(indexes_2, stride_2, 5, 3);

	index_t expected_out_2[5] = {0, 0, 0, 0, 0};
	for (int i = 0; i < 5; i++) {
		arbiter_assert(out_2[i] == expected_out_2[i]);
	}

	index_t indexes_3[5]      = {0, 0, 2, 2, 1};
	index_t stride_3[3]       = {0, 2, 2};

	index_t* out_3            = pascal_tensor_utils_get_convolution_start_index(indexes_3, stride_3, 5, 3);

	index_t expected_out_3[5] = {0, 0, 0, 4, 2};
	for (int i = 0; i < 5; i++) {
		arbiter_assert(out_3[i] == expected_out_3[i]);
	}

	free(out_1);
	free(out_2);
	free(out_3);
}

static void test_pascal_tensor_utils_get_convolution_array() {
	index_t ndim                  = 4;
	index_t shape[4]              = {2, 2, 5, 5};
	double  values[100]           = {0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101, 0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101};

	Tensor a                      = pascal_tensor_new(values, shape, ndim);

	index_t filter_ndim           = 3;
	index_t filter_shape[3]       = {2, 2, 2};
	double  filter_values[8]      = {1, -1, 0, 1, -1, 1, -1, -1};

	Tensor filter                 = pascal_tensor_new(filter_values, filter_shape, filter_ndim);

	index_t start_index[4]        = {0, 0, 0, 0};

	double* out_values            = pascal_tensor_utils_get_convolution_array(a, filter->shape, filter->size, filter->ndim, start_index);

	double expected_out_values[8] = {0.86828343, -0.46120553, -0.97302879, 0.07627669, 0.84440266, -0.62857164, -0.84286261, 0.89234251};
	for (int i = 0; i < 8; i++) {
		arbiter_assert(out_values[i] == expected_out_values[i]);
	}

	index_t start_index_2[4]        = {1, 0, 2, 3};

	double* out_values_2            = pascal_tensor_utils_get_convolution_array(a, filter->shape, filter->size, filter->ndim, start_index_2);

	double expected_out_values_2[8] = {-0.02316006, 0.38237511, 0.07150826, 0.90584821, -0.0158001, 0.77235944, -0.16560254, 0.09952113};
	for (int i = 0; i < 8; i++) {
		arbiter_assert(out_values_2[i] == expected_out_values_2[i]);
	}

	free(out_values);
	free(out_values_2);

	pascal_tensor_free(a);
	pascal_tensor_free(filter);
}

#define NUM_TESTS 8

int main() {
	void (*tests[NUM_TESTS])() = {
			test_pascal_tensor_utils_apply_transpose_map,
			test_get_size_from_shape,
			test_pascal_tensor_utils_shapes_equal,
			test_get_stride,
			test_pascal_tensor_utils_index_from_linear_index_transpose_safe,
			test_pascal_tensor_utils_index_from_linear_index,
			test_pascal_tensor_utils_get_convolution_start_index,
			test_pascal_tensor_utils_get_convolution_array,
	};
	arbiter_run_tests(NUM_TESTS, "pascal_tensor_utils", tests);
}
