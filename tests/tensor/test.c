#include "arbiter.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
	#include <gsl/gsl_randist.h>
	#include <gsl/gsl_rng.h>
#endif

static void test_pascal_tensor_free() {
	double  repeated_value = 3.0;
	index_t shape[3]       = {2, 3, 4};
	index_t ndim           = (&shape)[1] - shape;

	Tensor tensor          = pascal_tensor_new_repeat(repeated_value, shape, 3);

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_get() {
	double  values[48] = {0.17470035, -0.08200542, 0.57812676, -0.51472315, -0.66347028, -0.51001588, -0.71936854, 0.08425302, 0.47787879, -0.5885576, 0.75566958, -0.17169499, 0.52061366, -0.8806104, 0.87073119, 0.44958004, 0.41029544, 0.99975808, -0.71216538, -0.99701229, -0.36841142, -0.72472595, -0.53552446, -0.59361857, 0.31244563, -0.79451479, -0.37782478, -0.45297128, -0.36674374, 0.53225689, -0.32157635, 0.30953923, -0.08298934, 0.06163304, 0.85232222, -0.43796949, -0.77353275, -0.45054634, 0.19609561, -0.93332283, -0.26829233, -0.52319693, -0.48703697, 0.71508883, 0.14141374, -0.50283888, 0.74186519, 0.71052414};
	index_t shape[4]   = {2, 2, 3, 2};

	Tensor tensor      = pascal_tensor_new(values, shape, 4);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				for (int l = 0; l < shape[3]; l++) {
					index_t indexes[4]   = {i, j, k, l};
					double  value        = pascal_tensor_get(tensor, indexes);
					index_t linear_index = l + k * 2 + j * 6 + i * 12;

					arbiter_assert(fabs(value - values[linear_index]) < ARBITER_FLOATINGPOINT_ACCURACY);
				}
			}
		}
	}

	pascal_tensor_free(tensor);
}

static Tensor return_null() {
	return NULL;
}

static void test_pascal_tensor_null() {
	Tensor t = return_null();

	arbiter_assert(t == NULL);
}

static void test_pascal_tensor_init() {
	Tensor tensor = pascal_tensor_init();

	arbiter_assert(tensor->size == 0);
	arbiter_assert(tensor->ndim == 0);
	arbiter_assert(tensor->shape == NULL);
	arbiter_assert(tensor->_stride == NULL);
	arbiter_assert(tensor->values == NULL);
	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_new() {
	double  values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t ndim      = 2;
	index_t shape[2]  = {3, 2};

	Tensor tensor     = pascal_tensor_new(values, shape, ndim);

	arbiter_assert(tensor->size == 6);
	arbiter_assert(tensor->ndim == ndim);

	index_t stride[2] = {2, 1};

	for (int i = 0; i < 2; i++) {
		arbiter_assert(tensor->shape[i] == shape[i]);
		arbiter_assert(tensor->_stride[i] == stride[i]);
	}

	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	for (int i = 0; i < tensor->size; i++) {
		arbiter_assert(tensor->values[i] == values[i]);
		arbiter_assert(&(tensor->values[i]) != &values[i]);
	}

	arbiter_assert(tensor->values != values);
	arbiter_assert(tensor->shape != shape);
	arbiter_assert(&(tensor->ndim) != &ndim);

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_linspace() {
	double  start  = 0.0;
	double  end    = 11.0;
	index_t ndim   = 5;
	Tensor  tensor = pascal_tensor_linspace(start, end, ndim);

	arbiter_assert(tensor->size == 5);
	arbiter_assert(tensor->ndim == 2);

	index_t expected_shape[2]  = {5, 1};
	index_t expected_stride[2] = {1, 1};
	for (int i = 0; i < 2; i++) {
		arbiter_assert(tensor->shape[i] == expected_shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	double expected_values[5] = {0.0, 2.75, 5.5, 8.25, 11.0};
	for (int i = 0; i < tensor->size; i++) {
		arbiter_assert(fabs(tensor->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_eye() {
	index_t n      = 3;
	Tensor  tensor = pascal_tensor_eye(n);

	arbiter_assert(tensor->size == 9);
	arbiter_assert(tensor->ndim == 2);

	index_t expected_shape[2]  = {3, 3};
	index_t expected_stride[2] = {3, 1};
	for (int i = 0; i < 2; i++) {
		arbiter_assert(tensor->shape[i] == expected_shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	double expected_values[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
	for (int i = 0; i < 9; i++) {
		arbiter_assert(fabs(tensor->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_new_repeat() {
	double  repeated_value = 3.0;
	index_t shape[3]       = {2, 3, 4};
	index_t ndim           = 3;
	Tensor  tensor         = pascal_tensor_new_repeat(repeated_value, shape, ndim);

	index_t expected_size  = 24;
	index_t expected_ndim  = 3;

	arbiter_assert(tensor->size == expected_size);
	arbiter_assert(tensor->ndim == expected_ndim);

	index_t expected_stride[3] = {12, 4, 1};

	for (int i = 0; i < 3; i++) {
		arbiter_assert(tensor->shape[i] == shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(tensor, (index_t[]){i, j, k}) - repeated_value) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_zeros() {
	index_t shape[3]      = {2, 3, 4};
	index_t ndim          = 3;

	Tensor tensor         = pascal_tensor_zeros(shape, ndim);

	index_t expected_size = 24;
	index_t expected_ndim = 3;

	arbiter_assert(tensor->size == expected_size);
	arbiter_assert(tensor->ndim == expected_ndim);

	index_t expected_stride[3] = {12, 4, 1};

	for (int i = 0; i < 3; i++) {
		arbiter_assert(tensor->shape[i] == shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(tensor, (index_t[]){i, j, k}) - 0) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_ones() {
	index_t shape[3]      = {2, 3, 4};
	index_t ndim          = 3;
	Tensor  tensor        = pascal_tensor_ones(shape, ndim);

	index_t expected_size = 24;
	index_t expected_ndim = 3;

	arbiter_assert(tensor->size == expected_size);
	arbiter_assert(tensor->ndim == expected_ndim);

	index_t expected_stride[3] = {12, 4, 1};

	for (int i = 0; i < 3; i++) {
		arbiter_assert(tensor->shape[i] == shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(tensor, (index_t[]){i, j, k}) - 1) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_copy() {
	double  values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape[2]  = {3, 2};
	index_t ndim      = 2;

	Tensor a          = pascal_tensor_new(values, shape, 2);
	Tensor b          = pascal_tensor_copy(a);

	arbiter_assert(a->size == b->size);
	arbiter_assert(a->ndim == b->ndim);

	for (int i = 0; i < 2; i++) {
		arbiter_assert(a->shape[i] == b->shape[i]);
		arbiter_assert(a->_stride[i] == b->_stride[i]);
	}

	arbiter_assert(a->_transpose_map == b->_transpose_map);
	arbiter_assert(a->_transpose_map_inverse == b->_transpose_map_inverse);

	for (int i = 0; i < 6; i++) {
		arbiter_assert(a->values[i] == b->values[i]);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_reshape() {
	index_t ndim          = 1;
	index_t shape[1]      = {24};

	index_t new_ndim      = 3;
	index_t new_shape[3]  = {4, 3, 2};

	double values[24]     = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor b              = pascal_tensor_reshape(a, new_shape, new_ndim);

	index_t expected_size = 24;
	index_t expected_ndim = 3;

	arbiter_assert(b->size == expected_size);
	arbiter_assert(b->ndim == expected_ndim);

	index_t* expected_stride = (index_t[]){6, 2, 1};

	for (int i = 0; i < new_ndim; i++) {
		arbiter_assert(b->shape[i] == new_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(b->_transpose_map == NULL);
	arbiter_assert(b->_transpose_map_inverse == NULL);

	double expected_values[24] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0};
	for (int i = 0; i < b->size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_transpose() {
	double  repeated_value                    = 3.0;
	index_t shape[3]                          = {2, 3, 4};
	index_t ndim                              = 3;

	double values[24]                         = {-0.63405041, -0.11258597, 0.74705898, -0.28961194, -0.70223178, 0.71885991, -0.84620906, 0.93766196, 0.080927, -0.97218154, 0.79848487, -0.60933186, -0.25850228, 0.01016925, -0.16950571, -0.16315033, -0.61483418, 0.14335652, -0.44128188, -0.51254281, 0.56902254, 0.41054392, -0.59765366, 0.56065721};

	Tensor a                                  = pascal_tensor_new(values, shape, 3);

	index_t transpose_map[3]                  = {2, 1, 0};
	Tensor  b                                 = pascal_tensor_transpose(a, transpose_map);

	index_t transposed_shape[3]               = {4, 3, 2};
	index_t expected_stride[3]                = {1, 4, 12};
	index_t expected_transpose_map[3]         = {2, 1, 0};

	index_t expected_transpose_map_inverse[3] = {2, 1, 0};
	for (int i = 0; i < ndim; i++) {
		arbiter_assert(b->shape[i] == transposed_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
		arbiter_assert(b->_transpose_map[i] == expected_transpose_map[i]);
		arbiter_assert(b->_transpose_map_inverse[i] == expected_transpose_map_inverse[i]);
		arbiter_assert(transposed_shape[b->_transpose_map_inverse[i]] == shape[i]);
	}

	double transposed_values[24] = {-0.63405041, -0.25850228, -0.70223178, -0.61483418, 0.080927, 0.56902254, -0.11258597, 0.01016925, 0.71885991, 0.14335652, -0.97218154, 0.41054392, 0.74705898, -0.16950571, -0.84620906, -0.44128188, 0.79848487, -0.59765366, -0.28961194, -0.16315033, 0.93766196, -0.51254281, -0.60933186, 0.56065721};

	Tensor expected_tensor       = pascal_tensor_new(transposed_values, transposed_shape, 3);

	for (int i = 0; i < transposed_shape[0]; i++) {
		for (int j = 0; j < transposed_shape[1]; j++) {
			for (int k = 0; k < transposed_shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(b, (index_t[]){i, j, k}) - pascal_tensor_get(expected_tensor, (index_t[]){i, j, k})) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	index_t transpose_map2[3]                  = {0, 2, 1};
	Tensor  c                                  = pascal_tensor_transpose(b, transpose_map2);

	index_t transposed_shape2[3]               = {4, 2, 3};
	index_t expected_stride2[3]                = {1, 12, 4};
	index_t expected_transpose_map2[3]         = {2, 0, 1};
	index_t expected_transpose_map_inverse2[3] = {1, 2, 0};
	for (int i = 0; i < ndim; i++) {
		arbiter_assert(c->shape[i] == transposed_shape2[i]);
		arbiter_assert(c->_stride[i] == expected_stride2[i]);
		arbiter_assert(c->_transpose_map[i] == expected_transpose_map2[i]);
		arbiter_assert(c->_transpose_map_inverse[i] == expected_transpose_map_inverse2[i]);
		arbiter_assert(transposed_shape2[c->_transpose_map_inverse[i]] == shape[i]);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
	pascal_tensor_free(expected_tensor);
}

static void test_pascal_tensor_tile() {
	double  values[4]         = {1.0, 2.0, 3.0, 4.0};
	index_t shape[3]          = {2, 1, 2};
	index_t ndim              = 3;
	index_t tile_map[3]       = {1, 3, 2};

	Tensor tensor             = pascal_tensor_new(values, shape, 3);
	Tensor repeated_tensor    = pascal_tensor_tile(tensor, tile_map);

	index_t repeated_shape[3] = {2, 3, 4};
	for (int i = 0; i < tensor->ndim; i++) {
		arbiter_assert(repeated_tensor->shape[i] == repeated_shape[i]);
	}

	for (int i = 0; i < repeated_shape[0]; i++) {
		for (int j = 0; j < repeated_shape[1]; j++) {
			for (int k = 0; k < repeated_shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(tensor, (index_t[]){i % shape[0], j % shape[1], k % shape[2]}) - pascal_tensor_get(repeated_tensor, (index_t[]){i, j, k})) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(tensor);
	pascal_tensor_free(repeated_tensor);
}

static void test_pascal_tensor_append() {
	double  values1[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	index_t ndim1      = 3;
	index_t shape1[3]  = {2, 2, 2};

	double  values2[4] = {1.0, 2.0, 3.0, 4.0};
	index_t ndim2      = 3;
	index_t shape2[3]  = {2, 1, 2};

	Tensor a           = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b           = pascal_tensor_new(values2, shape2, ndim2);

	Tensor c           = pascal_tensor_append(a, b, 1);

	arbiter_assert(c->ndim == 3);
	arbiter_assert(c->size == 12);

	index_t expected_shape[3]  = {2, 3, 2};
	index_t expected_stride[3] = {6, 2, 1};

	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[12] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double  values3[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
	index_t ndim3       = 3;
	index_t shape3[3]   = {4, 2, 2};

	Tensor d            = pascal_tensor_new(values3, shape3, ndim3);
	Tensor e            = pascal_tensor_append(a, d, 0);

	arbiter_assert(e->ndim == 3);
	arbiter_assert(e->size == 24);

	index_t expected_shape2[3]  = {6, 2, 2};
	index_t expected_stride2[3] = {4, 2, 1};

	for (int i = 0; i < e->ndim; i++) {
		arbiter_assert(e->shape[i] == expected_shape2[i]);
		arbiter_assert(e->_stride[i] == expected_stride2[i]);
	}

	arbiter_assert(e->_transpose_map == NULL);
	arbiter_assert(e->_transpose_map_inverse == NULL);

	double expected_values2[24] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
	for (int i = 0; i < e->size; i++) {
		arbiter_assert(fabs(e->values[i] - expected_values2[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
	pascal_tensor_free(d);
	pascal_tensor_free(e);
}

static void test_pascal_tensor_expand_dims() {
	index_t ndim       = 3;
	index_t shape[3]   = {3, 5, 2};
	double  values[30] = {-0.71353715, 0.74392175, -0.77815781, -0.04787376, -0.79798544, -0.6806299, 0.39037488, 0.03146592, 0.84952967, -0.00957007, 0.78675469, 0.06932357, -0.04877769, 0.9643257, -0.68393041, 0.28116961, 0.82492452, 0.41888289, 0.68680333, -0.57874896, 0.83991778, -0.82101937, 0.72221873, -0.5357723, -0.02127044, -0.79242291, -0.29979601, -0.02517566, 0.63775995, 0.83682423};

	Tensor a           = pascal_tensor_new(values, shape, ndim);
	Tensor b           = pascal_tensor_expand_dims(a, 1);

	arbiter_assert(b->ndim == 4);
	arbiter_assert(b->size == a->size);

	index_t expected_shape[4]  = {3, 1, 5, 2};
	index_t expected_stride[4] = {10, 10, 2, 1};

	for (int i = 0; i < ndim + 1; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(b->_transpose_map == NULL);
	arbiter_assert(b->_transpose_map_inverse == NULL);

	for (int i = 0; i < a->shape[0]; i++) {
		for (int j = 0; j < a->shape[1]; j++) {
			for (int k = 0; k < a->shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(b, (index_t[]){i, 0, j, k}) - pascal_tensor_get(a, (index_t[]){i, j, k})) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_diag() {
	double  values[18] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
	index_t shape[3]   = {2, 3, 3};
	index_t ndim       = 3;

	Tensor tensor      = pascal_tensor_new(values, shape, 3);
	Tensor diag        = pascal_tensor_diag(tensor);

	arbiter_assert(diag->ndim == 2);
	arbiter_assert(diag->size == 6);

	index_t expected_shape[2]  = {2, 3};
	index_t expected_stride[2] = {3, 1};
	for (int i = 0; i < diag->ndim; i++) {
		arbiter_assert(diag->shape[i] == expected_shape[i]);
		arbiter_assert(diag->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(diag->_transpose_map == NULL);
	arbiter_assert(diag->_transpose_map_inverse == NULL);

	double expected_values[6] = {1.0, 5.0, 30.0, 40.0, 12.0, 16.0};
	for (int i = 0; i < diag->size; i++) {
		arbiter_assert(fabs(diag->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(tensor);
	pascal_tensor_free(diag);
}

static void test_pascal_tensor_clamp() {
	index_t ndim       = 3;
	index_t shape[3]   = {3, 2, 2};
	double  values[12] = {0.03992382, -4.45962422, -0.42867344, 1.76058905, -1.31076798, -1.29633849, -1.8428066, 2.91669712, 4.10416207, -4.38678569, 0.76771607, -2.7582438};

	Tensor a           = pascal_tensor_new(values, shape, ndim);

	double clamp_min   = -3.0;
	double clamp_max   = 3.0;

	Tensor b           = pascal_tensor_clamp(a, clamp_min, clamp_max);

	arbiter_assert(b->ndim == 3);
	arbiter_assert(b->size == 12);
	for (int i = 0; i < b->ndim; i++) {
		arbiter_assert(b->shape[i] == shape[i]);
		arbiter_assert(b->_stride[i] == a->_stride[i]);
	}

	double expected_values[12] = {0.03992382, -3.0, -0.42867344, 1.76058905, -1.31076798, -1.29633849, -1.8428066, 2.91669712, 3.0, -3.0, 0.76771607, -2.7582438};
	for (int i = 0; i < b->size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_flatten() {
	index_t ndim              = 3;
	index_t shape[3]          = {3, 5, 2};
	double  values[30]        = {-0.89444761, -0.03248509, -0.81345225, 0.51545551, -0.11966749, -0.9749127, 0.25895287, 0.67283844, -0.20833743, 0.39180246, -0.69752343, -0.64137352, -0.29197714, -0.58979983, -0.70701457, -0.28647029, 0.78943045, -0.35522369, -0.99543363, -0.64252159, 0.87524469, 0.31856941, 0.10481295, 0.8457841, -0.72429679, -0.05641867, -0.94359539, 0.486787, -0.80175021, -0.62255077};

	Tensor a                  = pascal_tensor_new(values, shape, ndim);
	Tensor b                  = pascal_tensor_flatten(a);

	index_t transpose_map[3]  = {1, 2, 0};

	Tensor a_t                = pascal_tensor_transpose(a, transpose_map);
	Tensor b_t                = pascal_tensor_flatten(a_t);

	index_t expected_ndim     = 1;
	index_t expected_size     = 30;
	index_t expected_shape[1] = {30};

	arbiter_assert(b->ndim == expected_ndim);
	arbiter_assert(b->size == expected_size);
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
	}

	double expected_values[30] = {-0.89444761, -0.03248509, -0.81345225, 0.51545551, -0.11966749, -0.9749127, 0.25895287, 0.67283844, -0.20833743, 0.39180246, -0.69752343, -0.64137352, -0.29197714, -0.58979983, -0.70701457, -0.28647029, 0.78943045, -0.35522369, -0.99543363, -0.64252159, 0.87524469, 0.31856941, 0.10481295, 0.8457841, -0.72429679, -0.05641867, -0.94359539, 0.486787, -0.80175021, -0.62255077};

	for (int i = 0; i < b->size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	arbiter_assert(b_t->ndim == expected_ndim);
	arbiter_assert(b_t->size == expected_size);
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(b_t->shape[i] == expected_shape[i]);
	}

	double expected_values_t[30] = {-0.89444761, -0.69752343, 0.87524469, -0.03248509, -0.64137352, 0.31856941, -0.81345225, -0.29197714, 0.10481295, 0.51545551, -0.58979983, 0.8457841, -0.11966749, -0.70701457, -0.72429679, -0.9749127, -0.28647029, -0.05641867, 0.25895287, 0.78943045, -0.94359539, 0.67283844, -0.35522369, 0.486787, -0.20833743, -0.99543363, -0.80175021, 0.39180246, -0.64252159, -0.62255077};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(b_t->values[i] - expected_values_t[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(a_t);
	pascal_tensor_free(b_t);
}

static void test_pascal_tensor_add() {
	index_t ndim          = 4;
	index_t shape[4]      = {1, 2, 3, 4};
	double  values_a[24]  = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor a              = pascal_tensor_new(values_a, shape, ndim);

	double values_b[24]   = {0.72845861, 0.97162978, 0.77802518, -0.22238461, 0.25692836, 0.93356073, 0.38697885, 0.27001943, -0.30481988, -0.52725457, 0.6657026, -0.447067, -0.87401265, 0.10793704, -0.51434179, -0.56570821, 0.19829446, -0.16639044, -0.61645527, 0.42522851, -0.79327276, 0.39967739, -0.9563691, 0.91056665};
	Tensor b              = pascal_tensor_new(values_b, shape, ndim);
	Tensor c              = pascal_tensor_add(a, b);

	index_t expected_size = 24;
	index_t expected_ndim = 4;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);

	index_t expected_shape[4]  = {1, 2, 3, 4};
	index_t expected_stride[4] = {24, 12, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[24] = {0.30516595, 1.66906988, 1.73020603, -1.08961028, 0.54412323, 0.00502852, 0.83766387, -0.25063282, -0.30963148, 0.44079085, -0.19240904, -0.60796246, -0.07170111, 0.90661917, -0.57555176, 0.22126302, 1.15371453, -0.15393932, -0.62438353, 0.63744219, -0.31334407, 0.17349932, -0.31400088, 1.83009924};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_add_broadcast() {
	index_t ndim          = 4;
	index_t shape[4]      = {1, 2, 3, 4};
	double  values_a[24]  = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor a              = pascal_tensor_new(values_a, shape, ndim);

	double values_b[24]   = {0.72845861};
	Tensor b              = pascal_tensor_new(values_b, (index_t[]){1}, 1);
	Tensor c              = pascal_tensor_add(a, b);

	index_t expected_size = 24;
	index_t expected_ndim = 4;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);

	index_t expected_shape[4]  = {1, 2, 3, 4};
	index_t expected_stride[4] = {24, 12, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[24] = {0.30516595, 1.42589871, 1.68063946, -0.13876706, 1.01565348, -0.2000736, 1.17914363, 0.20780636, 0.723647, 1.69650403, -0.12965303, 0.56756314, 1.53077014, 1.52714074, 0.66724864, 1.51542984, 1.68387868, 0.74090974, 0.72053035, 0.94067229, 1.2083873, 0.50228054, 1.37082683, 1.6479912};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_subtract() {
	index_t ndim          = 3;
	index_t shape[3]      = {2, 3, 4};
	double  values_a[24]  = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor a              = pascal_tensor_new(values_a, shape, ndim);

	double values_b[24]   = {0.72845861, 0.97162978, 0.77802518, -0.22238461, 0.25692836, 0.93356073, 0.38697885, 0.27001943, -0.30481988, -0.52725457, 0.6657026, -0.447067, -0.87401265, 0.10793704, -0.51434179, -0.56570821, 0.19829446, -0.16639044, -0.61645527, 0.42522851, -0.79327276, 0.39967739, -0.9563691, 0.91056665};
	Tensor b              = pascal_tensor_new(values_b, shape, ndim);
	Tensor c              = pascal_tensor_subtract(a, b);

	index_t expected_size = 24;
	index_t expected_ndim = 3;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);

	index_t expected_shape[3]  = {2, 3, 4};
	index_t expected_stride[3] = {12, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[24] = {-1.15175127, -0.27418969, 0.17415567, -0.64484106, 0.03026651, -1.86209294, 0.06370617, -0.79067168, 0.30000827, 1.49529999, -1.52381424, 0.28617153, 1.67632418, 0.69074509, 0.45313182, 1.35267943, 0.75712562, 0.17884157, 0.60852701, -0.21301483, 1.27320145, -0.62585545, 1.59873732, 0.00896594};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_scalar_multiply() {
	double  repeated_value = 5.3;
	double  k              = -3.3;
	index_t shape[3]       = {2, 3, 4};

	Tensor a               = pascal_tensor_new_repeat(repeated_value, shape, 3);
	Tensor b               = pascal_tensor_scalar_multiply(a, k);

	index_t expected_size  = 24;
	index_t expected_ndim  = 3;

	arbiter_assert(b->size == expected_size);
	arbiter_assert(b->ndim == expected_ndim);

	index_t expected_shape[3]  = {2, 3, 4};
	index_t expected_stride[3] = {12, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(b->_transpose_map == NULL);
	arbiter_assert(b->_transpose_map_inverse == NULL);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(b, (index_t[]){i, j, k}) - -17.49) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_multiply() {
	index_t ndim          = 3;
	index_t shape[3]      = {2, 3, 4};
	double  values_a[24]  = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor a              = pascal_tensor_new(values_a, shape, ndim);

	double values_b[24]   = {0.72845861, 0.97162978, 0.77802518, -0.22238461, 0.25692836, 0.93356073, 0.38697885, 0.27001943, -0.30481988, -0.52725457, 0.6657026, -0.447067, -0.87401265, 0.10793704, -0.51434179, -0.56570821, 0.19829446, -0.16639044, -0.61645527, 0.42522851, -0.79327276, 0.39967739, -0.9563691, 0.91056665};
	Tensor b              = pascal_tensor_new(values_b, shape, ndim);
	Tensor c              = pascal_tensor_multiply(a, b);

	index_t expected_size = 24;
	index_t expected_ndim = 3;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);

	index_t expected_shape[3]  = {2, 3, 4};
	index_t expected_stride[3] = {12, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[24] = {-0.30835118, 0.67765357, 0.74082068, 0.19285765, 0.07378851, -0.86684121, 0.17440557, -0.14058623, 0.00146667, -0.51040637, -0.57124715, 0.07193105, -0.70123042, 0.08620739, 0.03148285, -0.44519608, 0.1894545, -0.00207175, 0.00488742, 0.09023931, -0.38071436, -0.09039826, -0.61434112, 0.83729571};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_divide() {
	index_t ndim          = 3;
	index_t shape[3]      = {2, 3, 4};
	double  values_a[24]  = {-0.42329266, 0.6974401, 0.95218085, -0.86722567, 0.28719487, -0.92853221, 0.45068502, -0.52065225, -0.00481161, 0.96804542, -0.85811164, -0.16089547, 0.80231153, 0.79868213, -0.06120997, 0.78697123, 0.95542007, 0.01245113, -0.00792826, 0.21221368, 0.47992869, -0.22617807, 0.64236822, 0.91953259};

	Tensor a              = pascal_tensor_new(values_a, shape, ndim);

	double values_b[24]   = {0.72845861, 0.97162978, 0.77802518, -0.22238461, 0.25692836, 0.93356073, 0.38697885, 0.27001943, -0.30481988, -0.52725457, 0.6657026, -0.447067, -0.87401265, 0.10793704, -0.51434179, -0.56570821, 0.19829446, -0.16639044, -0.61645527, 0.42522851, -0.79327276, 0.39967739, -0.9563691, 0.91056665};
	Tensor b              = pascal_tensor_new(values_b, shape, ndim);
	Tensor c              = pascal_tensor_divide(a, b);

	index_t expected_size = 24;
	index_t expected_ndim = 3;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);

	index_t expected_shape[3]  = {2, 3, 4};
	index_t expected_stride[3] = {12, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[24] = {-0.58107991, 0.71780437, 1.22384323, 3.89966585, 1.11780136, -0.99461361, 1.16462442, -1.92820291, 0.01578509, -1.83601144, -1.28903153, 0.35989118, -0.91796329, 7.39951855, 0.11900641, -1.3911257, 4.81818842, -0.0748308, 0.01286105, 0.49905798, -0.60499832, -0.56590159, -0.67167396, 1.00984655};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_reciprocal() {
	double  values[12]    = {0.07150023, 0.8126589, -0.9965845, -0.6709441, -0.01702797, -0.40626801, -0.56654643, -0.89393956, -0.6045496, -0.93168161, -0.39074094, -0.85622983};
	index_t ndim          = 3;
	index_t shape[3]      = {3, 2, 2};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor b              = pascal_tensor_reciprocal(a);

	index_t expected_ndim = 3;
	arbiter_assert(expected_ndim == b->ndim);

	index_t expected_shape[3] = {3, 2, 2};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(expected_shape[i] == b->shape[i]);
	}

	double expected_values[12] = {13.98596899, 1.23052858, -1.00342721, -1.49043714, -58.7269063782, -2.46142934, -1.76508041, -1.11864386, -1.65412399, -1.07332804, -2.5592404, -1.16791073};
	for (int i = 0; i < b->size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static double operation_pascal_tensor_map(double a) {
	return exp(a);
}

static void test_pascal_tensor_map() {
	double  values[4]     = {0.1, 0.2, 0.3, 0.4};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 1, 2};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor b              = pascal_tensor_map(a, operation_pascal_tensor_map);

	index_t expected_size = 4;
	index_t expected_ndim = 3;

	arbiter_assert(b->size == expected_size);
	arbiter_assert(b->ndim == expected_ndim);

	index_t expected_shape[3]  = {2, 1, 2};
	index_t expected_stride[3] = {2, 2, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(b->_transpose_map == NULL);
	arbiter_assert(b->_transpose_map_inverse == NULL);

	double expected_values[4] = {1.10517092, 1.22140276, 1.34985881, 1.4918247};
	for (int i = 0; i < b->size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_square() {
	double  values[4]     = {0.1, 0.2, 0.3, 0.4};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 1, 2};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor b              = pascal_tensor_square(a);

	index_t expected_size = 4;
	index_t expected_ndim = 3;

	arbiter_assert(b->size == expected_size);
	arbiter_assert(b->ndim == expected_ndim);

	index_t expected_shape[3]  = {2, 1, 2};
	index_t expected_stride[3] = {2, 2, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(b->_transpose_map == NULL);
	arbiter_assert(b->_transpose_map_inverse == NULL);

	double expected_values[4] = {0.01, 0.04, 0.09, 0.16};
	for (int i = 0; i < b->size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_sum() {
	double  values[240]   = {0.78570881, 0.84403863, -0.26894181, 0.22054765, -0.9237383, 0.40055429, 0.84111446, 0.84045328, 0.32300506, -0.27624813, -0.67944788, 0.10941423, -0.63671027, -0.71541513, 0.240755, 0.44857613, -0.42720178, -0.6199451, -0.12198761, 0.03168983, 0.64320446, -0.3710589, 0.21699936, 0.43580219, -0.28702099, 0.1376122, 0.38242398, 0.70358061, 0.39856418, -0.34288356, -0.28732142, 0.91609906, -0.66467226, -0.28191155, 0.45250333, -0.35851501, -0.96366255, -0.27088465, 0.34333765, -0.12474236, 0.59772242, -0.42133908, 0.66076816, -0.34625112, 0.28251039, 0.5227484, 0.1235704, -0.21758213, 0.17095902, 0.61755615, 0.51473943, -0.36308426, -0.17300956, 0.14983142, -0.33547086, -0.70904009, -0.2533716, 0.72886371, -0.24898706, 0.70991861, 0.04893437, 0.13905885, -0.26870298, -0.98974455, 0.10614501, 0.21476867, 0.25326855, 0.46069209, 0.98717726, 0.46152699, -0.90752142, 0.67865356, -0.49145951, 0.24011225, -0.47556022, -0.71377398, 0.57757255, -0.44500762, -0.87793224, -0.76169721, 0.30504478, -0.47131141, -0.437951, -0.15442129, -0.57325682, -0.78285999, 0.19800772, -0.71372706, -0.76435117, -0.7117422, -0.63766315, -0.35676664, -0.04686275, -0.90586105, 0.9493131, 0.88964297, 0.66510156, 0.42981278, 0.50596967, 0.12121635, -0.96523139, -0.05414058, -0.79184744, -0.52890563, -0.78294503, 0.83467466, 0.47923744, 0.55179352, -0.80127359, 0.94917705, -0.00335658, -0.64764054, 0.32389428, -0.87146541, -0.17957602, 0.67471877, 0.13065678, -0.47210521, 0.25076306, 0.18629568, 0.97589329, 0.23013299, -0.70336469, -0.12827187, -0.4211749, -0.03294918, 0.91115413, 0.80470566, 0.68398502, -0.41775507, -0.0129372, 0.42426213, -0.83829894, 0.06892077, -0.15105143, 0.86573491, -0.64049636, 0.38670159, -0.12544035, -0.20750955, 0.47159886, 0.85569868, 0.95909449, 0.18450371, -0.13573082, 0.78920518, -0.16799613, 0.87340655, 0.70342144, -0.60714961, -0.7033921, -0.51700685, 0.44102305, 0.96111023, -0.61104567, -0.88530371, -0.4506001, -0.53628358, 0.53955612, 0.90050575, 0.04781273, 0.32451334, -0.21987123, 0.09754333, 0.77988031, -0.2907648, -0.00403675, 0.95858171, 0.61130973, 0.20881563, 0.38000419, 0.82974313, -0.2058397, -0.66676845, 0.05103404, -0.60752532, 0.8022679, 0.09197716, 0.31903346, 0.52357622, 0.94045315, -0.56922651, -0.8095668, -0.50183362, 0.24263121, 0.58945911, -0.8340168, -0.55776915, 0.17539351, -0.33533543, -0.80651966, 0.35477878, 0.63897225, 0.27701651, -0.5908882, -0.53113482, -0.50337106, -0.02507717, 0.05609085, 0.25372773, -0.94125712, -0.38736448, -0.09310707, 0.55038985, -0.14751978, -0.03974561, 0.80693146, 0.8770375, -0.35508231, 0.29548506, 0.2172682, -0.83226948, -0.87309827, -0.46077382, 0.88296455, 0.52799379, -0.29680631, 0.26102891, 0.66282009, 0.49592629, 0.5588142, -0.24524281, -0.01928812, 0.991894, -0.05624001, 0.28813116, -0.45522423, 0.92966448, 0.77894844, 0.19047419, -0.7831458, -0.60652266, -0.81043961, 0.6459, -0.21361733, -0.7533639, -0.21542849, -0.15516718, 0.40141772, 0.31847885};
	index_t ndim          = 5;
	index_t shape[5]      = {3, 2, 4, 5, 2};

	Tensor a              = pascal_tensor_new(values, shape, ndim);

	index_t axes[3]       = {0, 2, 3};
	Tensor  b             = pascal_tensor_sum(a, axes, 3);

	index_t expected_ndim = 2;
	index_t expected_size = 4;

	arbiter_assert(b->ndim == expected_ndim);
	arbiter_assert(b->size == expected_size);

	index_t expected_shape[2]  = {2, 2};
	index_t expected_stride[2] = {2, 1};

	for (int i = 0; i < b->ndim; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(b->_transpose_map == NULL);
	arbiter_assert(b->_transpose_map_inverse == NULL);

	double expected_values[4] = {-1.738642, 0.11744584, 0.06446001, 6.86082353};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_sum_mask() {
	double  values[240]   = {0.78570881, 0.84403863, -0.26894181, 0.22054765, -0.9237383, 0.40055429, 0.84111446, 0.84045328, 0.32300506, -0.27624813, -0.67944788, 0.10941423, -0.63671027, -0.71541513, 0.240755, 0.44857613, -0.42720178, -0.6199451, -0.12198761, 0.03168983, 0.64320446, -0.3710589, 0.21699936, 0.43580219, -0.28702099, 0.1376122, 0.38242398, 0.70358061, 0.39856418, -0.34288356, -0.28732142, 0.91609906, -0.66467226, -0.28191155, 0.45250333, -0.35851501, -0.96366255, -0.27088465, 0.34333765, -0.12474236, 0.59772242, -0.42133908, 0.66076816, -0.34625112, 0.28251039, 0.5227484, 0.1235704, -0.21758213, 0.17095902, 0.61755615, 0.51473943, -0.36308426, -0.17300956, 0.14983142, -0.33547086, -0.70904009, -0.2533716, 0.72886371, -0.24898706, 0.70991861, 0.04893437, 0.13905885, -0.26870298, -0.98974455, 0.10614501, 0.21476867, 0.25326855, 0.46069209, 0.98717726, 0.46152699, -0.90752142, 0.67865356, -0.49145951, 0.24011225, -0.47556022, -0.71377398, 0.57757255, -0.44500762, -0.87793224, -0.76169721, 0.30504478, -0.47131141, -0.437951, -0.15442129, -0.57325682, -0.78285999, 0.19800772, -0.71372706, -0.76435117, -0.7117422, -0.63766315, -0.35676664, -0.04686275, -0.90586105, 0.9493131, 0.88964297, 0.66510156, 0.42981278, 0.50596967, 0.12121635, -0.96523139, -0.05414058, -0.79184744, -0.52890563, -0.78294503, 0.83467466, 0.47923744, 0.55179352, -0.80127359, 0.94917705, -0.00335658, -0.64764054, 0.32389428, -0.87146541, -0.17957602, 0.67471877, 0.13065678, -0.47210521, 0.25076306, 0.18629568, 0.97589329, 0.23013299, -0.70336469, -0.12827187, -0.4211749, -0.03294918, 0.91115413, 0.80470566, 0.68398502, -0.41775507, -0.0129372, 0.42426213, -0.83829894, 0.06892077, -0.15105143, 0.86573491, -0.64049636, 0.38670159, -0.12544035, -0.20750955, 0.47159886, 0.85569868, 0.95909449, 0.18450371, -0.13573082, 0.78920518, -0.16799613, 0.87340655, 0.70342144, -0.60714961, -0.7033921, -0.51700685, 0.44102305, 0.96111023, -0.61104567, -0.88530371, -0.4506001, -0.53628358, 0.53955612, 0.90050575, 0.04781273, 0.32451334, -0.21987123, 0.09754333, 0.77988031, -0.2907648, -0.00403675, 0.95858171, 0.61130973, 0.20881563, 0.38000419, 0.82974313, -0.2058397, -0.66676845, 0.05103404, -0.60752532, 0.8022679, 0.09197716, 0.31903346, 0.52357622, 0.94045315, -0.56922651, -0.8095668, -0.50183362, 0.24263121, 0.58945911, -0.8340168, -0.55776915, 0.17539351, -0.33533543, -0.80651966, 0.35477878, 0.63897225, 0.27701651, -0.5908882, -0.53113482, -0.50337106, -0.02507717, 0.05609085, 0.25372773, -0.94125712, -0.38736448, -0.09310707, 0.55038985, -0.14751978, -0.03974561, 0.80693146, 0.8770375, -0.35508231, 0.29548506, 0.2172682, -0.83226948, -0.87309827, -0.46077382, 0.88296455, 0.52799379, -0.29680631, 0.26102891, 0.66282009, 0.49592629, 0.5588142, -0.24524281, -0.01928812, 0.991894, -0.05624001, 0.28813116, -0.45522423, 0.92966448, 0.77894844, 0.19047419, -0.7831458, -0.60652266, -0.81043961, 0.6459, -0.21361733, -0.7533639, -0.21542849, -0.15516718, 0.40141772, 0.31847885};
	index_t ndim          = 5;
	index_t shape[5]      = {3, 2, 4, 5, 2};

	Tensor a              = pascal_tensor_new(values, shape, ndim);

	bool   axes_mask[5]   = {true, false, true, true, false};
	Tensor b              = pascal_tensor_sum_mask(a, axes_mask);

	index_t expected_ndim = 2;
	index_t expected_size = 4;

	arbiter_assert(b->ndim == expected_ndim);
	arbiter_assert(b->size == expected_size);

	index_t expected_shape[2]  = {2, 2};
	index_t expected_stride[2] = {2, 1};

	for (int i = 0; i < b->ndim; i++) {
		arbiter_assert(b->shape[i] == expected_shape[i]);
		arbiter_assert(b->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(b->_transpose_map == NULL);
	arbiter_assert(b->_transpose_map_inverse == NULL);

	double expected_values[4] = {-1.738642, 0.11744584, 0.06446001, 6.86082353};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(b->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
}

static void test_pascal_tensor_sum_all() {
	double  values[48]    = {0.6176742, 0.97099486, -0.92477063, -0.57681528, 0.65435152, 0.43493424, 0.27029481, 0.49376086, 0.01719251, 0.49211045, 0.82565774, -0.53168095, -0.73382348, 0.87631719, -0.9924667, 0.17018514, -0.41607983, 0.73156697, -0.69285031, 0.25877695, -0.99548799, 0.23783478, -0.94969453, -0.13338413, 0.6176742, 0.97099486, -0.92477063, -0.57681528, 0.65435152, 0.43493424, 0.27029481, 0.49376086, 0.01719251, 0.49211045, 0.82565774, -0.53168095, -0.73382348, 0.87631719, -0.9924667, 0.17018514, -0.41607983, 0.73156697, -0.69285031, 0.25877695, -0.99548799, 0.23783478, -0.94969453, -0.13338413};
	index_t ndim          = 4;
	index_t shape[4]      = {2, 2, 3, 4};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor sum            = pascal_tensor_sum_all(a);

	index_t expected_ndim = 1;
	index_t expected_size = 1;

	arbiter_assert(sum->ndim == expected_ndim);
	arbiter_assert(sum->size == expected_size);

	index_t expected_shape[1]  = {1};
	index_t expected_stride[1] = {1};

	for (int i = 0; i < sum->ndim; i++) {
		arbiter_assert(sum->shape[i] == expected_shape[i]);
		arbiter_assert(sum->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(sum->_transpose_map == NULL);
	arbiter_assert(sum->_transpose_map_inverse == NULL);

	for (int i = 0; i < sum->size; i++) {
		arbiter_assert(fabs(sum->values[0] - 0.2091968) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(sum);
}

static void test_pascal_tensor_prod_all() {
	double  values[4]         = {1, 2, 3, 4};
	index_t ndim              = 3;
	index_t shape[3]          = {2, 1, 2};

	Tensor a                  = pascal_tensor_new(values, shape, ndim);
	Tensor prod               = pascal_tensor_prod_all(a);

	index_t expected_ndim     = 1;
	index_t expected_size     = 1;
	index_t expected_shape[1] = {1};

	arbiter_assert(prod->ndim == expected_ndim);
	arbiter_assert(prod->size == expected_size);
	for (int i = 0; i < prod->ndim; i++) {
		arbiter_assert(prod->shape[i] == expected_shape[i]);
	}

	for (int i = 0; i < prod->size; i++) {
		arbiter_assert(fabs(prod->values[0] - 24.0) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(prod);
}

static void test_pascal_tensor_mean_all() {
	double  values[4]     = {0.1, 0.2, 0.3, 0.4};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 1, 2};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor mean           = pascal_tensor_mean_all(a);

	index_t expected_ndim = 1;
	index_t expected_size = 1;

	arbiter_assert(mean->ndim == expected_ndim);
	arbiter_assert(mean->size == expected_size);

	index_t expected_shape[1]  = {1};
	index_t expected_stride[1] = {1};

	for (int i = 0; i < mean->ndim; i++) {
		arbiter_assert(mean->shape[i] == expected_shape[i]);
		arbiter_assert(mean->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(mean->_transpose_map == NULL);
	arbiter_assert(mean->_transpose_map_inverse == NULL);

	for (int i = 0; i < mean->size; i++) {
		arbiter_assert(fabs(mean->values[0] - 0.25) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(mean);
}

static void test_pascal_tensor_dot() {
	double  values1[10]   = {-0.6394606222465, -0.9610495170248, -0.0735629470033, 0.4498678583843, -0.1595927908245, -0.0291458036644, -0.9744383708188, -0.0252567853603, 0.8836133046867, 0.7015901787536};
	index_t shape1[2]     = {10, 1};
	index_t ndim1         = 2;

	double  values2[10]   = {0.4599289404416, -0.7825278562621, 0.7878083405702, 0.7143084941457, -0.6698267648095, 0.2646680276470, -0.9590327744175, -0.7665254624400, -0.3672653767749, -0.6841753866501};
	index_t shape2[2]     = {10, 1};
	index_t ndim2         = 2;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);

	Tensor c              = pascal_tensor_dot(a, b);

	index_t expected_size = 1;
	index_t expected_ndim = 2;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);

	index_t expected_shape[2]  = {1, 1};
	index_t expected_stride[2] = {1, 1};

	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[1] = {0.9698650598064};

	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}
static void test_pascal_tensor_dot_broadcast() {
	double  values1[16]   = {0.5426412865335, -0.9584961012812, 0.2672964698526, 0.4976077650772, -0.6617783268749, -0.8233203716520, 0.3707196367356, 0.9067866923899, -0.6394606222465, -0.9610495170248, -0.0735629470033, 0.4498678583843, -0.1595927908245, -0.0291458036644, -0.9744383708188, -0.0252567853603};
	index_t shape1[5]     = {2, 1, 2, 4, 1};
	index_t ndim1         = 5;

	double  values2[8]    = {-0.0029859753948, -0.5504067089383, -0.6038742704808, 0.5210614243979, -0.9921034673442, 0.0243845267716, 0.6252419233042, 0.2250521336588};
	index_t shape2[4]     = {2, 1, 4, 1};
	index_t ndim2         = 4;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);

	Tensor c              = pascal_tensor_dot(a, b);

	index_t expected_size = 8;
	index_t expected_ndim = 5;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);

	index_t expected_shape[5]  = {2, 2, 2, 1, 1};
	index_t expected_stride[5] = {4, 2, 1, 1, 1};

	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);

	double expected_values[8] = {0.6238131212348, 0.7037606253388, -0.2826161275773, 1.0723400336366, 0.8097090735129, 0.591796509772, 0.6662254458097, -0.4573219600365};

	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_matmul() {
	double  values1[72]     = {0.5426412865335, -0.9584961012812, 0.2672964698526, 0.4976077650772, -0.0029859753948, -0.5504067089383, -0.6038742704808, 0.5210614243979, -0.6617783268749, -0.8233203716520, 0.3707196367356, 0.9067866923899, -0.9921034673442, 0.0243845267716, 0.6252419233042, 0.2250521336588, 0.4435106348636, -0.4162478636587, 0.8355482450259, 0.4291515667954, 0.0850887360225, -0.7156599047969, -0.2533184798971, 0.3482672301327, -0.1163336511540, -0.1319720133334, 0.2355339569386, 0.0262764851088, 0.3007943638629, 0.2020779068091, 0.6104463936655, 0.0432943047873, 0.8172977616173, -0.3615278220229, -0.8190813014582, -0.3985998867276, -0.7720312762729, 0.6573626526154, -0.9062073612215, 0.2525742966228, 0.0951723118385, 0.6385739913401, -0.6021049206424, 0.7137006049155, -0.2966947211358, 0.5092953830597, -0.4080765862406, 0.7678729591224, -0.3489767243355, -0.6699682045617, -0.2149415121068, -0.8130792508827, 0.6422113156739, -0.6976959607149, -0.2317711026156, 0.8885214244776, 0.9752509498037, -0.0873909058104, 0.6522456876855, -0.4972517315859, 0.1947432964618, 0.8056635206633, 0.0691158976036, 0.1804027259708, -0.9214364655492, -0.2856364827309, -0.8407738196881, -0.3890801633144, -0.3385613760357, 0.5476605924212, -0.9200815826200, -0.1410156431367};
	index_t shape1[4]       = {2, 3, 4, 3};
	index_t ndim1           = 4;

	double values2[72]      = {-0.3701462563146, 0.2729822861351, -0.3073056998399, -0.9138052875900, 0.7598303490358, 0.5264811742874, 0.7561932854497, -0.1649817123215, 0.2111551287875, 0.0269332548166, 0.1956732959259, -0.4755686777361, -0.3982573821186, -0.9492004358998, -0.3938748786979, -0.5158482491929, 0.1151563773253, 0.1310140397763, -0.0497355051699, -0.4144040474210, -0.8714978786104, 0.9576382915153, -0.3205843127243, -0.0099027382351, 0.9541614518454, -0.1184523501987, -0.3634543890421, 0.0395939717508, 0.1562728597649, 0.7078675010010, -0.8638054529241, -0.0709383844413, 0.5638982372383, 0.4372056207645, 0.1720439601064, -0.9258111735312, -0.2986872174337, 0.1263813689855, -0.4005402551509, 0.0246683065471, 0.3469338505694, -0.6816125332438, -0.8990446596920, -0.3243682258706, -0.7838724544411, -0.6421943828578, 0.7716541923354, -0.2692700575718, -0.5624613016409, 0.5049923404372, -0.7862408312129, 0.4892064815511, -0.0604294131190, 0.1965113425582, -0.7047596154294, -0.6319303558137, 0.2901442529365, -0.9027439874732, -0.5027749843945, 0.0848170324560, -0.5464533134600, -0.2371769301907, 0.8444655738071, 0.8507137457356, 0.1334998491500, 0.0669417699780, -0.9702799507335, 0.9557985268040, 0.1460578080664, 0.5835139925532, 0.1231147205527, 0.7546704831299};

	index_t shape2[4]       = {2, 3, 3, 4};
	index_t ndim2           = 4;

	Tensor a                = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b                = pascal_tensor_new(values2, shape2, ndim2);
	Tensor c                = pascal_tensor_matmul(a, b);
	Tensor d                = pascal_tensor_matmul(b, a);

	index_t expected_ndim   = 4;

	index_t expected_size_c = 96;
	index_t expected_size_d = 54;

	arbiter_assert(c->size == expected_size_c);
	arbiter_assert(c->ndim == expected_ndim);
	arbiter_assert(d->size == expected_size_d);
	arbiter_assert(d->ndim == expected_ndim);

	index_t expected_shape_c[4]  = {2, 3, 4, 4};
	index_t expected_stride_c[4] = {48, 16, 4, 1};

	index_t expected_shape_d[4]  = {2, 3, 3, 3};
	index_t expected_stride_d[4] = {27, 9, 3, 1};

	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape_c[i]);
		arbiter_assert(c->_stride[i] == expected_stride_c[i]);
		arbiter_assert(d->shape[i] == expected_shape_d[i]);
		arbiter_assert(d->_stride[i] == expected_stride_d[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);
	arbiter_assert(d->_transpose_map == NULL);
	arbiter_assert(d->_transpose_map_inverse == NULL);

	double expected_values_c[96] = {-8.7271004740194e-01, -3.4929953006907e-01, -8.3926229499337e-01, -4.6485197758743e-01, -3.0267768559426e-01, 1.1944180133251e-01, -2.6287557187463e-01, -1.9246778475296e-01, 4.7970219663249e-01, 9.1658207389056e-02, 4.5010480935001e-01, 7.8057893924936e-01, 7.7790564510418e-01, -5.1522505447913e-03, 7.1078068388852e-01, 2.5995320024778e-01, -1.4697644631682e-01, 1.5436553662379e+00, 1.8910910382978e-01, 4.9547818295908e-01, 3.2420353457966e-01, -5.5412835641288e-01, 2.2741927975257e-02, -2.9576335765201e-01, -3.5749836987943e-01, -6.5539364615986e-01, -3.7772364758322e-01, -6.0970085702949e-01, -4.7668550532969e-02, 9.7963045144298e-01, 1.8283037014095e-01, 4.7069931304710e-01, 1.1924537113281e-03, 2.3338064960469e-02, 1.9680231554723e-01, -2.1330419895991e-01, 1.8602938005727e-01, 2.9815963993600e-01, -2.3461183221759e-01, -2.0738345986004e-01, 1.0501029091491e+00, 3.1566499655470e-01, -1.1865613411374e-01, -7.3556463057833e-01, -6.9772586239180e-01, -7.1124732471241e-01, 7.7034906525225e-01, 4.1281821077993e-01, 1.1690082184474e+00, 3.6324284576847e-02, -9.8104748734378e-01, 1.1742246758565e-02, -5.4298277917386e-01, -4.4303858546690e-01, 3.0602796561120e-01, -1.9658914918004e-01, 6.6001876164335e-01, -3.7202643810076e-01, -6.2942718433475e-01, -1.6646370313367e-01, -8.9561006337585e-01, -1.5060813766275e-01, 7.5541816101487e-01, -6.1834662980122e-02, 1.7440764356692e-01, -1.1384976641720e-01, 8.5461349940755e-01, 2.3442086922863e-01, 2.1608468758573e-01, 3.4544384760465e-01, 5.3745558195772e-01, -8.6277296571229e-01, 3.5963290609148e-01, -8.2284002475258e-01, -9.3429789411227e-01, -5.9214969409177e-01, -1.3453555366488e-01, 5.3293294865653e-01, -1.4096039000566e-01, -4.9710146330485e-01, 1.1232755355075e-02, 4.8074018209479e-02, -6.0875596712606e-01, 9.8788253317604e-01, -2.6331273508758e-01, -2.7114283712714e-01, 1.0112291641494e+00, -9.4279795998595e-01, 3.5805196404951e-01, -2.4188861512173e-02, -3.7416975359340e-01, -1.3426423695898e+00, -4.4269813360511e-01, -2.7376894870082e-01, 1.3378561275876e+00, -5.1993057077222e-01};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values_c[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double expected_values_d[54] = {0.8729099788788, -0.1449220847549, -0.8744482916950, 0.3534835747513, -0.3970052970415, -0.7367143489331, 0.4013669991488, -0.2768166309059, -0.5191149183089, 0.2215633915774, -0.4690499327064, -0.0670719144334, 0.1702538950243, 0.1445462871099, -0.1310890003480, 0.8193579407542, 0.2684009897882, -0.9742388159531, -0.3502973403246, -0.2097185941782, -0.1120317919424, -0.5012402464449, 0.2130053584260, -0.4978582832218, 0.3856179983363, 0.8228533454578, 0.7308061522121, 0.5162472722335, -0.4802501952131, 0.4891567152158, -0.0638816159601, -0.3460909234445, -0.7319858299925, -0.1587825264651, 0.0842052696963, -0.1354485631234, -0.0748371844315, 0.3216339784010, -1.2414757209079, 0.0798770427973, -0.8716802736534, -0.4972054011668, 0.7418652893052, -1.1655446297797, 0.0349695918394, -0.3933076364839, -1.3330048810387, -0.3758895997968, 1.3773138462557, -0.4560225689676, 0.1838227214735, 0.4435029439303, -1.1622575310835, -0.3046802006861};
	for (int i = 0; i < d->size; i++) {
		arbiter_assert(fabs(d->values[i] - expected_values_d[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
	pascal_tensor_free(d);
}

static void test_pascal_tensor_matmul_broadcast() {
	double  values1[36]   = {0.5426412865335, -0.9584961012812, 0.2672964698526, 0.4976077650772, -0.0029859753948, -0.5504067089383, -0.6038742704808, 0.5210614243979, -0.6617783268749, -0.8233203716520, 0.3707196367356, 0.9067866923899, -0.9921034673442, 0.0243845267716, 0.6252419233042, 0.2250521336588, 0.4435106348636, -0.4162478636587, 0.8355482450259, 0.4291515667954, 0.0850887360225, -0.7156599047969, -0.2533184798971, 0.3482672301327, -0.1163336511540, -0.1319720133334, 0.2355339569386, 0.0262764851088, 0.3007943638629, 0.2020779068091, 0.6104463936655, 0.0432943047873, 0.8172977616173, -0.3615278220229, -0.8190813014582, -0.3985998867276};
	index_t shape1[5]     = {3, 1, 3, 2, 2};
	index_t ndim1         = 5;

	double  values2[8]    = {-0.7720312762729, 0.6573626526154, -0.9062073612215, 0.2525742966228, 0.0951723118385, 0.6385739913401, -0.6021049206424, 0.7137006049155};
	index_t shape2[4]     = {2, 1, 2, 2};
	index_t ndim2         = 4;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);
	Tensor c              = pascal_tensor_matmul(a, b);
	Tensor d              = pascal_tensor_matmul(b, a);

	index_t expected_ndim = 5;
	index_t expected_size = 72;

	arbiter_assert(c->size == expected_size);
	arbiter_assert(c->ndim == expected_ndim);
	arbiter_assert(d->size == expected_size);
	arbiter_assert(d->ndim == expected_ndim);

	index_t expected_shape[5]  = {3, 2, 3, 2, 2};
	index_t expected_stride[5] = {24, 12, 4, 2, 1};
	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
		arbiter_assert(c->_stride[i] == expected_stride[i]);
		arbiter_assert(d->shape[i] == expected_shape[i]);
		arbiter_assert(d->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(c->_transpose_map == NULL);
	arbiter_assert(c->_transpose_map_inverse == NULL);
	arbiter_assert(d->_transpose_map == NULL);
	arbiter_assert(d->_transpose_map_inverse == NULL);

	// pascal_tensor_print(pascal_tensor_flatten(c));
	double expected_values_c[72] = {0.4496601776823, 0.1146206369375, -0.6572970544775, 0.3013936477154, 0.5010878777006, -0.1409814560727, -0.0059798746903, -0.2653576695248, 1.2570125477416, -0.6429779201631, -1.1079439299898, 0.4727282547984, 0.6287596447363, -0.3375626350859, -0.2741728609207, 0.5258315365711, 0.3311184056249, -0.3947323673447, -0.3712057579683, -0.0137365493776, 0.4327422737435, -1.0101986748622, -0.5106984845889, 0.8839061289864, 0.7438374684282, -0.6460128622644, -0.6866502202081, 0.4678530735919, 0.0348027966659, 0.1864138160127, -1.0339696869046, 0.6576508658662, 0.5828451083902, -0.1248231398558, -0.1200325382800, -0.0785587572141, -0.1091028241268, -0.6161282194569, -0.0759992777778, 0.5598830744473, 0.2928348193607, -0.0138619957814, -0.1788732119372, 0.8398451106057, 0.4390004419034, -0.4564314531873, -0.2338023183246, 0.0867959400299, 0.2094072271319, -0.1098061359526, -0.2056515256135, 0.1614679914582, -0.4153471433194, 0.2487706660960, -0.5105173260949, 0.4122196891988, -0.3033602603929, 0.4459483891964, 0.9935705340706, -0.6391095430583, 0.0683892560902, -0.1684761496926, 0.0065951102171, 0.1691594022956, -0.0930448070460, 0.3363025818343, 0.0320298805904, 0.4207143616182, 0.2954617980227, 0.2638824684778, 0.1620450921227, -0.8075249961809};
	double expected_values_d[72] = {-0.2432253285438, 1.0670977287879, -0.4242333104966, 0.9942791539415, -0.3946591258944, 0.7674575140886, -0.1498172062321, 0.6303893340702, 0.7546108100881, 1.2317167827737, 0.6933426428037, 0.9751299924519, 0.2223329993633, 0.2265370868201, -0.1359573365400, 0.9322581819442, -0.3859025853499, 0.2803527945701, -0.4291875616569, 0.7032844415972, 0.1737489448051, 0.5006930942997, 0.6630428159781, 1.1428994579247, 1.1769455953181, 0.1291152502339, 1.0569715041983, 0.0347449467020, 0.2068541292640, 0.6034645817490, -0.1908745918116, 0.4855995332682, -0.2322132733695, 0.7814496998968, -0.1410897758222, 0.7364996245456, 0.3048424499474, 0.1460331710360, 1.0435859183550, 0.1459378003731, 0.5757693102284, 0.2344297574162, 0.3292913523024, 0.5569106197381, -0.1536645010694, 0.1542833875692, -0.2320258989892, 0.6794608830026, 0.2446444438881, 0.1191597018386, 0.1649122345381, 0.1262307747020, 0.1690620039906, -0.1275504052623, -0.1183989982406, -0.1721894581112, -1.1694128910306, 0.0170861069731, -0.9475201314802, 0.2269430875794, 0.1393341164535, 0.0042193983674, 0.2381457913414, 0.0982145419323, 0.4184424850990, 0.0468788385728, 0.2545661938442, -0.0907729305268, -0.4452598984707, -0.2889429592312, -1.0766778242255, -0.0668032996876};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values_c[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
		arbiter_assert(fabs(d->values[i] - expected_values_d[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double  values3[3]     = {0.5426412865335, -0.9584961012812, 0.2672964698526};
	index_t shape3[2]      = {3, 1};
	index_t ndim3          = 2;

	double  values4[4]     = {0.4976077650772, -0.0029859753948, -0.5504067089383, -0.6038742704808};
	index_t shape4[2]      = {1, 4};
	index_t ndim4          = 2;

	Tensor e               = pascal_tensor_new(values3, shape3, ndim3);
	Tensor f               = pascal_tensor_new(values4, shape4, ndim4);
	Tensor g               = pascal_tensor_matmul(e, f);

	index_t expected_size2 = 12;
	index_t expected_ndim2 = 2;

	arbiter_assert(g->size == expected_size2);
	arbiter_assert(g->ndim == expected_ndim2);

	index_t expected_shape2[2]  = {3, 4};
	index_t expected_stride2[2] = {4, 1};
	for (int i = 0; i < g->ndim; i++) {
		arbiter_assert(g->shape[i] == expected_shape2[i]);
		arbiter_assert(g->_stride[i] == expected_stride2[i]);
	}

	arbiter_assert(g->_transpose_map == NULL);
	arbiter_assert(g->_transpose_map_inverse == NULL);

	double expected_values_g[12] = {0.2700225178306, -0.0016203135298, -0.2986734046549, -0.3276871110381, -0.4769551027938, 0.0028620457745, 0.5275626846364, 0.5788111339198, 0.1330087989764, -0.0007981406821, -0.1471217702824, -0.1614134607343};

	for (int i = 0; i < g->size; i++) {
		arbiter_assert(fabs(g->values[i] - expected_values_g[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
	pascal_tensor_free(d);
	pascal_tensor_free(e);
	pascal_tensor_free(f);
	pascal_tensor_free(g);
}

static void test_pascal_tensor_linalg_solve() {
	index_t ndim_a        = 3;
	index_t shape_a[3]    = {1, 5, 5};
	double  values_a[25]  = {6.80, -2.11, 5.66, 5.97, 8.23, -6.05, -3.30, 5.36, -4.44, 1.08, -0.45, 2.58, -2.70, .27, .04, 8.32, 2.71, 4.35, 7.17, 2.14, -9.67, -5.14, -7.26, 6.08, -6.87};

	Tensor a              = pascal_tensor_new(values_a, shape_a, ndim_a);

	index_t ndim_y        = 2;
	index_t shape_y[2]    = {5, 3};
	double  values_y[15]  = {4.02, 6.19, -8.22, -7.57, -3.03, -1.56, 4.00, -8.67, 1.75, 2.86, 9.81, -4.09, -4.57, -8.61, 8.99};

	Tensor y              = pascal_tensor_new(values_y, shape_y, ndim_y);

	Tensor x              = pascal_tensor_linalg_solve(a, y);

	index_t expected_size = 15;
	index_t expected_ndim = 3;

	arbiter_assert(x->size == expected_size);
	arbiter_assert(x->ndim == expected_ndim);

	index_t expected_shape[3]  = {1, 5, 3};
	index_t expected_stride[3] = {15, 3, 1};

	for (int i = 0; i < x->ndim; i++) {
		arbiter_assert(x->shape[i] == expected_shape[i]);
		arbiter_assert(x->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(x->_transpose_map == NULL);
	arbiter_assert(x->_transpose_map_inverse == NULL);

	double expected_values[15] = {0.18562296, 2.37078723, -0.36489311, 0.65241122, -2.0274866, 0.2347112, -0.85785922, 0.79693265, -0.3576818, 0.16763128, -0.53749583, 0.12510189, 0.97072612, -1.884707, -0.48187916};
	for (int i = 0; i < x->size; i++) {
		arbiter_assert(fabs(x->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	Tensor c = pascal_tensor_matmul(a, x);
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - values_y[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(y);
	pascal_tensor_free(x);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_linalg_inv() {
	index_t ndim          = 3;
	index_t shape[3]      = {3, 3, 3};
	double  values[27]    = {-0.73755023, -0.26606191, -0.52607579, 0.00322696, -0.39497577, -0.69726951, 0.08398574, 0.38625244, 0.97182299, 0.1643701, -0.8198551, -0.48212327, 0.38354449, 0.4542986, -0.78827282, -0.32870814, 0.96254244, 0.0334494, 0.22146974, -0.18732193, 0.84994723, -0.72536213, -0.23125245, -0.16282841, -0.77999073, 0.22726785, 0.13851565};

	Tensor a              = pascal_tensor_new(values, shape, ndim);

	Tensor a_inv          = pascal_tensor_linalg_inv(a);

	index_t expected_size = 27;
	index_t expected_ndim = 3;

	arbiter_assert(a_inv->size == expected_size);
	arbiter_assert(a_inv->ndim == ndim);

	index_t expected_shape[3]  = {3, 3, 3};
	index_t expected_stride[3] = {9, 3, 1};

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(a_inv->shape[i] == a->shape[i]);
		arbiter_assert(a_inv->_stride[i] == a->_stride[i]);
	}

	arbiter_assert(a_inv->_transpose_map == NULL);
	arbiter_assert(a_inv->_transpose_map_inverse == NULL);

	double expected_values[27] = {-1.38354971, 0.66887906, -0.26904388, -0.7453471, -8.12538337, -6.23332768, 0.41580677, 3.17164019, 3.52968999, -2.38364548, 1.34479838, -2.66500865, -0.7585191, 0.47115868, 0.17046263, -1.59694489, -0.34272812, -1.19845366, -0.01484942, -0.65419597, -0.67790513, -0.67917387, -2.07093716, 1.73305, 1.03072815, -0.28595579, 0.55859202};

	for (int i = 0; i < a->size; i++) {
		arbiter_assert(fabs(a_inv->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double eye[27]   = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
	Tensor t_eye     = pascal_tensor_matmul(a, a_inv);
	Tensor t_eye_com = pascal_tensor_matmul(a_inv, a);

	for (int i = 0; i < a->size; i++) {
		arbiter_assert(fabs(t_eye->values[i] - eye[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
		arbiter_assert(fabs(t_eye_com->values[i] - eye[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
		arbiter_assert(fabs(t_eye->values[i] - t_eye_com->values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_inv);
	pascal_tensor_free(t_eye);
	pascal_tensor_free(t_eye_com);
}

static void test_pascal_tensor_linalg_cholesky() {
	index_t ndim          = 3;
	index_t shape[3]      = {3, 3, 3};
	double  values[27]    = {114.20569479, 110.76900094, 148.28526411, 110.76900094,
	                         127.12550775, 149.16610585, 148.28526411, 149.16610585,
	                         202.96822899, 149.74072434, 51.18612863, 67.69083467,
	                         51.18612863, 84.69183898, 85.09284312, 67.69083467,
	                         85.09284312, 87.76867194, 59.76032128, 54.68930337,
	                         66.1149965, 54.68930337, 168.79003329, 159.23301018,
	                         66.1149965, 159.23301018, 160.22082355};

	Tensor a              = pascal_tensor_new(values, shape, ndim);

	Tensor a_chol         = pascal_tensor_linalg_cholesky(a);

	index_t expected_size = 27;
	index_t expected_ndim = 3;

	arbiter_assert(a_chol->size == expected_size);
	arbiter_assert(a_chol->ndim == ndim);

	index_t expected_shape[3]  = {3, 3, 3};
	index_t expected_stride[3] = {9, 3, 1};

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(a_chol->shape[i] == a->shape[i]);
		arbiter_assert(a_chol->_stride[i] == a->_stride[i]);
	}

	arbiter_assert(a_chol->_transpose_map == NULL);
	arbiter_assert(a_chol->_transpose_map_inverse == NULL);

	double expected_values[27] = {10.68670645, 0., 0., 10.36512058, 4.43731712,
	                              0., 13.87567487, 1.20412011, 2.99732727, 12.23685925,
	                              0., 0., 4.18294659, 8.19724325, 0.,
	                              5.53171637, 7.55790296, 0.21653809, 7.73048002, 0.,
	                              0., 7.07450291, 10.89685468, 0., 8.55250856,
	                              9.06025329, 2.23321095};

	for (int i = 0; i < a->size; i++) {
		arbiter_assert(fabs(a_chol->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	index_t transpose_map[3] = {0, 2, 1};
	Tensor  a_chol_T         = pascal_tensor_transpose(a_chol, transpose_map);

	Tensor a_from_chol       = pascal_tensor_matmul(a_chol, a_chol_T);

	arbiter_assert(a_from_chol->size == a->size);
	arbiter_assert(a_from_chol->ndim == a->ndim);

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(a_from_chol->shape[i] == a->shape[i]);
		arbiter_assert(a_from_chol->_stride[i] == a->_stride[i]);
	}

	for (int i = 0; i < a->size; i++) {
		arbiter_assert(fabs(a_from_chol->values[i] - a->values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(a_chol);
	pascal_tensor_free(a_chol_T);
	pascal_tensor_free(a_from_chol);
}

static void test_pascal_tensor_linalg_triangular_solve() {
	index_t ndim          = 3;
	index_t shape[3]      = {3, 5, 5};
	double  values[75]    = {0.69973001, 0., 0., 0., 0., 0.60849071, -0.09777915, 0., 0., 0., -0.13835329, 0.56052732, 0.20031312, 0., 0., -0.01646198, 0.43965995, -0.6403842, 0.96733269, 0., 0.68766794, 0.30913195, -0.92563211, 0.61447058, -0.32156378, 0.18165682, 0., 0., 0., 0., -0.73456942, 0.26073904, 0., 0., 0., 0.99116004, -0.32771097, -0.07552252, 0., 0., 0.35083807, -0.15702608, -0.74240568, -0.12882536, 0., -0.18272571, 0.2398248, -0.87147049, -0.68007153, -0.89609135, 0.04142994, 0., 0., 0., 0., -0.05803728, 0.81326461, 0., 0., 0., -0.93864523, 0.13530447, 0.7425683, 0., 0., 0.00248901, -0.55256774, 0.34356217, 0.99704562, 0., -0.00203851, 0.52824887, 0.22428514, 0.32503684, -0.34521337};

	Tensor a              = pascal_tensor_new(values, shape, ndim);

	index_t ndim_y        = 2;
	index_t shape_y[2]    = {5, 2};
	double  values_y[10]  = {0.20489036, -0.14240992, 0.75373731, -0.0327114, 0.64176891, -0.05575036, -0.94989752, 0.33446768, -0.47917708, -0.62754607};

	Tensor y              = pascal_tensor_new(values_y, shape_y, ndim_y);
	Tensor x              = pascal_tensor_linalg_triangular_solve(a, y, true);

	index_t expected_size = 30;
	index_t expected_ndim = 3;

	arbiter_assert(x->size == expected_size);
	arbiter_assert(x->ndim == expected_ndim);

	index_t expected_shape[3]  = {3, 5, 2};
	index_t expected_stride[3] = {10, 2, 1};

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(x->shape[i] == expected_shape[i]);
		arbiter_assert(x->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(x->_transpose_map == NULL);
	arbiter_assert(x->_transpose_map_inverse == NULL);

	Tensor c = pascal_tensor_matmul(a, x);
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - values_y[i % y->size]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	Tensor a_upper = pascal_tensor_transpose(a, (index_t[]){0, 2, 1});
	Tensor x_upper = pascal_tensor_linalg_triangular_solve(a_upper, y, false);

	Tensor c_upper = pascal_tensor_matmul(a_upper, x_upper);
	for (int i = 0; i < c_upper->size; i++) {
		arbiter_assert(fabs(c_upper->values[i] - values_y[i % y->size]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(y);
	pascal_tensor_free(x);
	pascal_tensor_free(c);
	pascal_tensor_free(a_upper);
	pascal_tensor_free(x_upper);
	pascal_tensor_free(c_upper);
}

static void test_pascal_tensor_convolution_2d() {
	index_t ndim             = 4;
	index_t shape[4]         = {2, 2, 5, 5};
	double  values[100]      = {0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101, 0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101};

	Tensor a                 = pascal_tensor_new(values, shape, ndim);

	index_t filter_ndim      = 3;
	index_t filter_shape[3]  = {2, 2, 2};
	double  filter_values[8] = {0.10579893, 0.67045058, -0.81264612, 0.42692948, 0.37382649, -0.35066912, -0.09350435, 0.8535618};

	Tensor filter            = pascal_tensor_new(filter_values, filter_shape, filter_ndim);

	index_t stride[3]        = {0, 1, 1};

	Tensor conv              = pascal_tensor_convolution(a, filter, stride);

	index_t expected_size    = 32;
	index_t expected_ndim    = ndim;

	arbiter_assert(conv->size == expected_size);
	arbiter_assert(conv->ndim == expected_ndim);

	index_t expected_shape[4]  = {2, 1, 4, 4};
	index_t expected_stride[4] = {16, 16, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(conv->shape[i] == expected_shape[i]);
		arbiter_assert(conv->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(conv->_transpose_map == NULL);
	arbiter_assert(conv->_transpose_map_inverse == NULL);

	double expected_values[32] = {1.98250234, -0.48924436, 0.03661835, 1.61710458, -0.11609795, -0.66694335, 0.05568384, 1.14154509, 1.31582533, 0.13343934, -0.58544471, 0.40621859, -1.74367143, 1.0754167, -1.26794906, 1.38282592, 1.98250234, -0.48924436, 0.03661835, 1.61710458, -0.11609795, -0.66694335, 0.05568384, 1.14154509, 1.31582533, 0.13343934, -0.58544471, 0.40621859, -1.74367143, 1.0754167, -1.26794906, 1.38282591};
	for (int i = 0; i < conv->size; i++) {
		arbiter_assert(fabs(conv->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(filter);
	pascal_tensor_free(conv);
}

static void test_pascal_tensor_convolution_3d() {
	index_t ndim              = 5;
	index_t shape[5]          = {1, 2, 2, 5, 5};
	double  values[100]       = {0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101, 0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101};

	Tensor a                  = pascal_tensor_new(values, shape, ndim);

	index_t filter_ndim       = 4;
	index_t filter_shape[4]   = {2, 2, 2, 2};
	double  filter_values[16] = {-0.15679321, 0.87743623, -0.70990645, 0.86646283, 0.93419727, 0.0009422, -0.93204437, -0.6509359, 0.70274056, 0.28413277, 0.896202, 0.38409922, 0.52464955, -0.07347934, -0.6287662, 0.84175161};

	Tensor filter             = pascal_tensor_new(filter_values, filter_shape, filter_ndim);

	index_t stride[4]         = {0, 1, 1, 1};

	Tensor conv               = pascal_tensor_convolution(a, filter, stride);

	index_t expected_size     = 16;
	index_t expected_ndim     = ndim;

	arbiter_assert(conv->size == expected_size);
	arbiter_assert(conv->ndim == expected_ndim);

	index_t expected_shape[5]  = {1, 1, 1, 4, 4};
	index_t expected_stride[5] = {16, 16, 16, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(conv->shape[i] == expected_shape[i]);
		arbiter_assert(conv->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(conv->_transpose_map == NULL);
	arbiter_assert(conv->_transpose_map_inverse == NULL);

	double expected_values[16] = {2.61570355, -3.2562472, -1.37798506, 1.20149169, -0.24564094, 0.63383654, -0.71122699, 0.90022137, 0.80842432, 0.08366585, 0.38815665, 1.77604167, -2.56488444, 3.32744815, -1.60405283, 3.25590574};
	for (int i = 0; i < conv->size; i++) {
		arbiter_assert(fabs(conv->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(filter);
	pascal_tensor_free(conv);
}

static void test_pascal_tensor_conv2d() {
	index_t ndim              = 4;
	index_t shape[4]          = {2, 2, 5, 5};
	double  values[100]       = {0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101, 0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101};

	Tensor a                  = pascal_tensor_new(values, shape, ndim);

	index_t filter_ndim       = 4;
	index_t filter_shape[4]   = {3, 2, 2, 2};
	double  filter_values[24] = {-0.70457698, 0.16246591, -0.14714709, -0.58620014, 0.24399399, 0.13618466, 0.77753431, 0.22491853, -0.54918535, 0.41332386, 0.53409409, 0.02459893, -0.42611966, -0.62349581, -0.4582492, 0.13642569, 0.45499252, 0.19117853, -0.57920068, 0.34400888, -0.26113195, -0.89959414, 0.07271935, -0.23780951};

	Tensor filter             = pascal_tensor_new(filter_values, filter_shape, filter_ndim);

	index_t stride[2]         = {1, 1};

	Tensor conv               = pascal_tensor_conv2d(a, filter, stride);

	// pascal_tensor_print(conv);
	index_t expected_size     = 96;
	index_t expected_ndim     = 4;

	arbiter_assert(conv->size == expected_size);
	arbiter_assert(conv->ndim == expected_ndim);

	index_t expected_shape[4]  = {2, 3, 4, 4};
	index_t expected_stride[4] = {48, 16, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(conv->shape[i] == expected_shape[i]);
		arbiter_assert(conv->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(conv->_transpose_map == NULL);
	arbiter_assert(conv->_transpose_map_inverse == NULL);

	double expected_values[96] = {-0.92246062, 0.98077338, 1.5256687, -0.89674089, -0.07182377, 0.53029387, 0.24707728, 0.81016273, 0.12401058, -0.74969302, 0.13780213, -0.46813875, 0.41528649, -0.45847713, 0.08040703, -0.96415047, -0.64521362, -0.2693019, 0.44409031, -0.35852005, 0.44736632, -0.39532985, -0.88371786, 0.83547993, 0.34341878, -1.21793581, 0.60502531, -0.15412671, 0.25848774, 1.47024937, -0.42411151, 0.77080457, 0.96816756, -0.66728201, 0.13654646, 0.91842066, -0.66471989, -1.47725316, 0.16285983, -0.70216421, 0.41117036, 0.61915862, -0.59717295, -0.39362724, -0.36234244, -0.5680956, 0.41561518, 0.32214676, -0.92246062, 0.98077338, 1.5256687, -0.89674089, -0.07182377, 0.53029387, 0.24707728, 0.81016273, 0.12401058, -0.74969302, 0.13780213, -0.46813875, 0.41528649, -0.45847713, 0.08040703, -0.96415047, -0.64521362, -0.2693019, 0.44409031, -0.35852005, 0.44736632, -0.39532985, -0.88371786, 0.83547993, 0.34341878, -1.21793581, 0.60502531, -0.15412671, 0.25848774, 1.47024937, -0.42411151, 0.77080457, 0.96816756, -0.66728201, 0.13654646, 0.91842066, -0.66471989, -1.47725316, 0.16285983, -0.70216421, 0.41117036, 0.61915862, -0.59717295, -0.39362724, -0.36234244, -0.5680956, 0.41561518, 0.32214676};
	for (int i = 0; i < conv->size; i++) {
		arbiter_assert(fabs(conv->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(filter);
	pascal_tensor_free(conv);
}

static void test_pascal_tensor_max_pool_2d() {
	index_t ndim            = 4;
	index_t shape[4]        = {2, 2, 5, 5};
	double  values[100]     = {0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101, 0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101};

	Tensor a                = pascal_tensor_new(values, shape, ndim);

	index_t filter_ndim     = 2;
	index_t filter_shape[2] = {2, 2};
	index_t stride[2]       = {1, 1};

	Tensor max_pooled       = pascal_tensor_max_pool(a, filter_shape, stride, filter_ndim);

	index_t expected_size   = 64;
	index_t expected_ndim   = ndim;

	arbiter_assert(max_pooled->size == expected_size);
	arbiter_assert(max_pooled->ndim == expected_ndim);

	index_t expected_shape[4]  = {2, 2, 4, 4};
	index_t expected_stride[4] = {32, 16, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(max_pooled->shape[i] == expected_shape[i]);
		arbiter_assert(max_pooled->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(max_pooled->_transpose_map == NULL);
	arbiter_assert(max_pooled->_transpose_map_inverse == NULL);

	double expected_values[64] = {0.86828343, 0.07627669, 0.48690214, 0.99700985, 0.8332759, 0.8332759, -0.02316006, 0.99700985, 0.8332759, 0.8332759, 0.60862678, 0.90584821, 0.74994097, 0.74994097, 0.60862678, 0.90584821, 0.89234251, 0.89234251, 0.61044636, 0.51678352, 0.89234251, 0.89234251, 0.61044636, 0.77235944, 0.27535274, 0.27535274, 0.19678562, 0.77235944, 0.83611266, 0.99119727, 0.99119727, 0.41878101, 0.86828343, 0.07627669, 0.48690214, 0.99700985, 0.8332759, 0.8332759, -0.02316006, 0.99700985, 0.8332759, 0.8332759, 0.60862678, 0.90584821, 0.74994097, 0.74994097, 0.60862678, 0.90584821, 0.89234251, 0.89234251, 0.61044636, 0.51678352, 0.89234251, 0.89234251, 0.61044636, 0.77235944, 0.27535274, 0.27535274, 0.19678562, 0.77235944, 0.83611266, 0.99119727, 0.99119727, 0.418781};

	for (int i = 0; i < max_pooled->size; i++) {
		arbiter_assert(fabs(max_pooled->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(max_pooled);
}

static void test_pascal_tensor_max_pool_3d() {
	index_t ndim            = 5;
	index_t shape[5]        = {1, 2, 2, 5, 5};
	double  values[100]     = {0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101, 0.86828343, -0.46120553, -0.61988487, 0.48690214, 0.40915901, -0.97302879, 0.07627669, -0.06117901, -0.98255291, 0.99700985, -0.16307585, 0.8332759, -0.56013273, -0.02316006, 0.38237511, -0.9902236, -0.46510756, 0.60862678, 0.07150826, 0.90584821, 0.51314706, 0.74994097, 0.5414147, -0.13191454, 0.73524396, 0.84440266, -0.62857164, 0.39747189, -0.70443326, 0.37575259, -0.84286261, 0.89234251, 0.61044636, -0.22612015, 0.51678352, -0.30051774, 0.05545741, 0.19678562, -0.0158001, 0.77235944, -0.69550929, 0.27535274, -0.16749314, -0.16560254, 0.09952113, 0.83611266, -0.93197992, 0.99119727, -0.9217425, 0.41878101};

	Tensor a                = pascal_tensor_new(values, shape, ndim);

	index_t filter_ndim     = 3;
	index_t filter_shape[3] = {2, 2, 2};
	index_t stride[3]       = {1, 1, 1};

	Tensor max_pooled       = pascal_tensor_max_pool(a, filter_shape, stride, filter_ndim);

	index_t expected_size   = 32;
	index_t expected_ndim   = ndim;

	arbiter_assert(max_pooled->size == expected_size);
	arbiter_assert(max_pooled->ndim == expected_ndim);

	index_t expected_shape[5]  = {1, 2, 1, 4, 4};
	index_t expected_stride[5] = {32, 16, 16, 4, 1};

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(max_pooled->shape[i] == expected_shape[i]);
		arbiter_assert(max_pooled->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(max_pooled->_transpose_map == NULL);
	arbiter_assert(max_pooled->_transpose_map_inverse == NULL);

	double expected_values[64] = {0.89234251, 0.89234251, 0.61044636, 0.99700985, 0.89234251, 0.89234251, 0.61044636, 0.99700985, 0.8332759, 0.8332759, 0.60862678, 0.90584821, 0.83611266, 0.99119727, 0.99119727, 0.90584821, 0.89234251, 0.89234251, 0.61044636, 0.99700985, 0.89234251, 0.89234251, 0.61044636, 0.99700985, 0.8332759, 0.8332759, 0.60862678, 0.90584821, 0.83611266, 0.99119727, 0.99119727, 0.90584821};

	for (int i = 0; i < max_pooled->size; i++) {
		arbiter_assert(fabs(max_pooled->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(max_pooled);
}

static void test_pascal_tensor_random_uniform() {
	index_t shape[5]      = {1, 5, 10, 3, 1};
	index_t ndim          = 5;

	Tensor tensor         = pascal_tensor_random_uniform(1.0, 0.01, shape, ndim);

	index_t expected_size = 150;
	index_t expected_ndim = 5;

	arbiter_assert(tensor->size == expected_size);
	arbiter_assert(tensor->ndim == expected_ndim);

	index_t expected_shape[5]  = {1, 5, 10, 3, 1};
	index_t expected_stride[5] = {150, 30, 3, 1, 1};

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(tensor->shape[i] == expected_shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(tensor->values[i] - 0.0) > ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_random_normal() {
	index_t shape[5]      = {1, 5, 10, 3, 1};
	index_t ndim          = 5;

	Tensor tensor         = pascal_tensor_random_normal(1.0, 1.0, shape, ndim);

	index_t expected_size = 150;
	index_t expected_ndim = 5;

	arbiter_assert(tensor->size == expected_size);
	arbiter_assert(tensor->ndim == expected_ndim);

	index_t expected_shape[5]  = {1, 5, 10, 3, 1};
	index_t expected_stride[5] = {150, 30, 3, 1, 1};

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(tensor->shape[i] == expected_shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	pascal_tensor_free(tensor);
}

static void test_pascal_tensor_uncertain_normal() {
#if TENSOR_BACKEND == TENSOR_BACKEND_GSL && !TENSOR_USE_UXHW
	const gsl_rng_type* T;
	gsl_rng*            r;

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
#endif

	index_t shape[5] = {1, 5, 10, 3, 1};
	index_t ndim     = 5;
	index_t size     = 150;

	double* means    = malloc(sizeof(double) * size);
	for (int i = 0; i < size; i++) {
		means[i] = 0.0;
	}

	double* stds = malloc(sizeof(double) * size);
	for (int i = 0; i < size; i++) {
		stds[i] = 1.0;
	}

#if TENSOR_USE_UXHW
	Tensor tensor = pascal_tensor_new(means, shape, ndim);
#elif TENSOR_BACKEND == TENSOR_BACKEND_GSL
	Tensor tensor = pascal_tensor_uncertain_normal(means, stds, shape, ndim, r);
#elif !TENSOR_USE_UXHW
	Tensor tensor = pascal_tensor_uncertain_normal(means, stds, shape, ndim);
#endif

	index_t expected_size = 150;
	index_t expected_ndim = 5;

	arbiter_assert(tensor->size == expected_size);
	arbiter_assert(tensor->ndim == expected_ndim);

	index_t expected_shape[5]  = {1, 5, 10, 3, 1};
	index_t expected_stride[5] = {150, 30, 3, 1, 1};

	for (int i = 0; i < ndim; i++) {
		arbiter_assert(tensor->shape[i] == expected_shape[i]);
		arbiter_assert(tensor->_stride[i] == expected_stride[i]);
	}

	arbiter_assert(tensor->_transpose_map == NULL);
	arbiter_assert(tensor->_transpose_map_inverse == NULL);

	pascal_tensor_free(tensor);
	free(means);
	free(stds);
}

#define NUM_TESTS 48

int main() {
	void (*tests[NUM_TESTS])() = {
			test_pascal_tensor_add,
			test_pascal_tensor_add_broadcast,
			test_pascal_tensor_append,
			test_pascal_tensor_clamp,
			test_pascal_tensor_conv2d,
			test_pascal_tensor_convolution_2d,
			test_pascal_tensor_convolution_3d,
			test_pascal_tensor_copy,
			test_pascal_tensor_diag,
			test_pascal_tensor_divide,
			test_pascal_tensor_dot,
			test_pascal_tensor_dot_broadcast,
			test_pascal_tensor_expand_dims,
			test_pascal_tensor_eye,
			test_pascal_tensor_flatten,
			test_pascal_tensor_free,
			test_pascal_tensor_get,
			test_pascal_tensor_init,
			test_pascal_tensor_linalg_inv,
			test_pascal_tensor_linalg_solve,
			test_pascal_tensor_linalg_cholesky,
			test_pascal_tensor_linalg_triangular_solve,
			test_pascal_tensor_linspace,
			test_pascal_tensor_map,
			test_pascal_tensor_matmul,
			test_pascal_tensor_matmul_broadcast,
			test_pascal_tensor_max_pool_2d,
			test_pascal_tensor_max_pool_3d,
			test_pascal_tensor_mean_all,
			test_pascal_tensor_multiply,
			test_pascal_tensor_new,
			test_pascal_tensor_new_repeat,
			test_pascal_tensor_ones,
			test_pascal_tensor_prod_all,
			test_pascal_tensor_random_normal,
			test_pascal_tensor_random_uniform,
			test_pascal_tensor_reciprocal,
			test_pascal_tensor_reshape,
			test_pascal_tensor_scalar_multiply,
			test_pascal_tensor_square,
			test_pascal_tensor_subtract,
			test_pascal_tensor_sum,
			test_pascal_tensor_sum_all,
			test_pascal_tensor_sum_mask,
			test_pascal_tensor_tile,
			test_pascal_tensor_transpose,
			test_pascal_tensor_uncertain_normal,
			test_pascal_tensor_zeros,
	};

	arbiter_run_tests(NUM_TESTS, "tensor", tests);
}
