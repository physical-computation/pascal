#include "arbiter.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static void test_add_after_transpose() {
	index_t ndim_a       = 3;
	index_t shape_a[3]   = {3, 5, 2};
	double  values_a[30] = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357, -0.04877769, 0.9643257, -0.68393041, 0.28116961, 0.82492452, 0.41888289, 0.68680333, -0.57874896, 0.83991778, -0.82101937, 0.72221873, -0.5357723, -0.02127044, -0.79242291, -0.29979601, -0.02517566, 0.63775995, 0.83682423};

	Tensor a             = pascal_tensor_new(values_a, shape_a, ndim_a);

	index_t ndim_b       = ndim_a;
	index_t shape_b[3]   = {2, 3, 5};
	double  values_b[30] = {-0.00794581, 0.39940027, 0.28077845, 0.93462348, 0.36752661, 0.56807948, -0.84993166, -0.45672162, -0.42752346, 0.87034361, 0.11870851, 0.76207276, 0.19297372, -0.16263325, -0.55326157, 0.29979912, 0.10017052, -0.05780453, 0.07166372, -0.16537522, -0.23256475, -0.40427211, -0.38698471, -0.42260267, 0.01371261, 0.10499274, -0.70052783, -0.58346512, 0.18664665, 0.05282858};

	Tensor b             = pascal_tensor_new(values_b, shape_b, ndim_b);
	Tensor a_t           = pascal_tensor_transpose(a, (index_t[]){2, 0, 1});

	Tensor c             = pascal_tensor_add(a_t, b);

	arbiter_assert(c->ndim == ndim_a);
	arbiter_assert(c->size == a->size);

	index_t expected_shape[3] = {2, 3, 5};
	for (int i = 0; i < ndim_a; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
	}

	double expected_values[30] = {-0.72148296, -0.37875754, -0.51720699, 1.32499836, 1.21705629, 1.35483417, -0.89870935, -1.14065203, 0.39740106, 1.55714695, 0.95862629, 1.48429149, 0.17170327, -0.46242926, 0.08449838, 1.04372087, 0.05229677, -0.73843443, 0.10312964, -0.17494529, -0.16324118, 0.56005358, -0.1058151, -0.00371978, -0.56503635, -0.71602663, -1.23630013, -1.37588803, 0.16147099, 0.88965281};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_matmul_after_transpose() {
	index_t ndim_a        = 3;
	index_t shape_a[3]    = {3, 2, 2};
	double  values_a[12]  = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357};

	Tensor a              = pascal_tensor_new(values_a, shape_a, ndim_a);

	index_t ndim_b        = 3;
	index_t shape_b[3]    = {1, 3, 2};
	double  values_b[6]   = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299};

	Tensor b              = pascal_tensor_new(values_b, shape_b, ndim_b);

	Tensor a_t            = pascal_tensor_transpose(a, (index_t[]){0, 2, 1});
	Tensor b_t            = pascal_tensor_transpose(b, (index_t[]){1, 2, 0});

	Tensor c              = pascal_tensor_matmul(a_t, b_t);

	index_t expected_ndim = 3;
	index_t expected_size = 6;

	arbiter_assert(c->ndim == expected_ndim);
	arbiter_assert(c->size == expected_size);

	index_t expected_shape[3] = {3, 2, 1};
	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
	}

	double expected_values[6] = {-0.06975326, -0.56643014, 0.60226989, 0.52813108, -1.21340107, -0.03954692};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(b);
	pascal_tensor_free(b_t);
	pascal_tensor_free(c);
}

static void test_linalg_solve_after_transpose() {
	index_t ndim_a        = 3;
	index_t shape_a[3]    = {3, 2, 2};
	double  values_a[12]  = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357};

	Tensor a              = pascal_tensor_new(values_a, shape_a, ndim_a);

	index_t ndim_b        = 3;
	index_t shape_b[3]    = {3, 2, 1};
	double  values_b[6]   = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299};

	Tensor b              = pascal_tensor_new(values_b, shape_b, ndim_b);
	Tensor a_t            = pascal_tensor_transpose(a, (index_t[]){0, 2, 1});

	Tensor c              = pascal_tensor_linalg_solve(a_t, b);

	index_t expected_ndim = 3;
	index_t expected_size = 6;

	arbiter_assert(c->ndim == expected_ndim);
	arbiter_assert(c->size == expected_size);

	index_t expected_shape[3] = {3, 2, 1};
	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
	}

	double expected_values[6] = {1.00000000e+00, -3.92416075e-18, -2.40936975e-02, -2.04261153e+00, 7.22910391e+00, -8.82018727e+00};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_linalg_inv_after_transpose() {
	index_t ndim_a        = 3;
	index_t shape_a[3]    = {3, 2, 2};
	double  values_a[12]  = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357};

	Tensor a              = pascal_tensor_new(values_a, shape_a, ndim_a);
	Tensor a_t            = pascal_tensor_transpose(a, (index_t[]){0, 2, 1});

	Tensor c              = pascal_tensor_linalg_inv(a_t);

	index_t expected_ndim = 3;
	index_t expected_size = 12;

	arbiter_assert(c->ndim == expected_ndim);
	arbiter_assert(c->size == expected_size);

	index_t expected_shape[3] = {3, 2, 2};
	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
	}

	double expected_values[12] = {-0.07809134, 1.2693256, -1.21347998, -1.16391683, 0.13078568, -1.62256326, 2.828986, -3.31676531, 1.04368816, -11.84483942, 0.14408042, 12.78993649};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(c);
}

static void test_expand_dims_after_transpose() {
	index_t ndim               = 3;
	index_t shape[3]           = {3, 5, 2};
	double  values[30]         = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357, -0.04877769, 0.9643257, -0.68393041, 0.28116961, 0.82492452, 0.41888289, 0.68680333, -0.57874896, 0.83991778, -0.82101937, 0.72221873, -0.5357723, -0.02127044, -0.79242291, -0.29979601, -0.02517566, 0.63775995, 0.83682423};

	Tensor a                   = pascal_tensor_new(values, shape, ndim);
	Tensor a_t                 = pascal_tensor_transpose(a, (index_t[]){2, 0, 1});
	Tensor b                   = pascal_tensor_expand_dims(a_t, 1);

	index_t expected_shape[4]  = {2, 1, 3, 5};
	index_t expected_stride[4] = {1, 1, 10, 2};

	for (int i = 0; i < ndim + 1; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	for (int i = 0; i < a_t->shape[0]; i++) {
		for (int j = 0; j < a_t->shape[1]; j++) {
			for (int k = 0; k < a_t->shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(b, (index_t[]){i, 0, j, k}) - pascal_tensor_get(a_t, (index_t[]){i, j, k})) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(b);
}

static void test_tile_after_transpose() {
	index_t ndim        = 3;
	index_t shape[3]    = {3, 5, 2};
	double  values[30]  = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357, -0.04877769, 0.9643257, -0.68393041, 0.28116961, 0.82492452, 0.41888289, 0.68680333, -0.57874896, 0.83991778, -0.82101937, 0.72221873, -0.5357723, -0.02127044, -0.79242291, -0.29979601, -0.02517566, 0.63775995, 0.83682423};

	Tensor a            = pascal_tensor_new(values, shape, ndim);
	Tensor a_t          = pascal_tensor_transpose(a, (index_t[]){2, 0, 1});

	index_t tile_map[3] = {3, 2, 4};
	Tensor  b           = pascal_tensor_tile(a_t, tile_map);

	arbiter_assert(b->ndim == 3);

	index_t expected_shape[3] = {9, 10, 8};
	index_t expected_size     = 720;

	arbiter_assert(b->size == expected_size);

	for (int i = 0; i < b->shape[0]; i++) {
		for (int j = 0; j < b->shape[1]; j++) {
			for (int k = 0; k < b->shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(b, (index_t[]){i, j, k}) - pascal_tensor_get(a_t, (index_t[]){i % a_t->shape[0], j % a_t->shape[1], k % a_t->shape[2]})) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(b);
}

static void test_reshape_after_transpose() {
	index_t ndim         = 3;
	index_t shape[3]     = {3, 2, 2};
	double  values[12]   = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};

	Tensor a             = pascal_tensor_new(values, shape, ndim);
	Tensor a_t           = pascal_tensor_transpose(a, (index_t[]){2, 0, 1});

	index_t new_ndim     = 2;
	index_t new_shape[2] = {3, 4};
	Tensor  b            = pascal_tensor_reshape(a_t, new_shape, new_ndim);

	arbiter_assert(b->ndim == new_ndim);

	index_t expected_size = 12;
	arbiter_assert(b->size == expected_size);

	double expected_values[12] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
	for (int i = 0; i < b->size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(b);
}

static void test_linalg_after_append() {
}

static void test_matmul_after_reshape_and_transpose() {
	index_t ndim_a        = 2;
	index_t shape_a[2]    = {2, 6};
	double  values_a[12]  = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357};

	Tensor a              = pascal_tensor_new(values_a, shape_a, ndim_a);

	index_t ndim_b        = 3;
	index_t shape_b[3]    = {1, 3, 2};
	double  values_b[6]   = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299};

	Tensor b              = pascal_tensor_new(values_b, shape_b, ndim_b);

	Tensor a_t            = pascal_tensor_transpose(a, (index_t[]){1, 0});
	Tensor a_r            = pascal_tensor_reshape(a_t, (index_t[]){3, 2, 2}, 3);
	Tensor b_t            = pascal_tensor_transpose(b, (index_t[]){1, 2, 0});

	Tensor c              = pascal_tensor_matmul(a_r, b_t);

	index_t expected_ndim = 3;
	index_t expected_size = 6;

	arbiter_assert(c->ndim == expected_ndim);
	arbiter_assert(c->size == expected_size);

	index_t expected_shape[3] = {3, 2, 1};
	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
	}

	double expected_values[6] = {0.79954363, -0.50740762, 0.5648594, 0.0377115, 0.101292, 0.49594906};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_t);
	pascal_tensor_free(a_r);
	pascal_tensor_free(b);
	pascal_tensor_free(b_t);
	pascal_tensor_free(c);
}

static void test_transpose_after_append() {
}

static void test_transpose_after_reshape() {
}

static void test_reshape_after_append() {
}

static void test_append_after_reshape() {
}

static void test_append_after_transpose() {
}

static void test_all_shape_operations() {
}

#define NUM_TESTS 15

int main() {
	void (*tests[NUM_TESTS])() = {
			test_add_after_transpose,
			test_all_shape_operations,
			test_append_after_reshape,
			test_append_after_transpose,
			test_expand_dims_after_transpose,
			test_linalg_after_append,
			test_linalg_inv_after_transpose,
			test_linalg_solve_after_transpose,
			test_matmul_after_reshape_and_transpose,
			test_matmul_after_transpose,
			test_reshape_after_append,
			test_reshape_after_transpose,
			test_tile_after_transpose,
			test_transpose_after_append,
			test_transpose_after_reshape,
	};

	arbiter_run_tests(NUM_TESTS, "pascal_tensor_integrations", tests);
}
