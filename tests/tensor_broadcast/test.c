#include "arbiter.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static void test_pascal_tensor_broadcast() {
	double  repeated_value    = 0.0;
	index_t shape1[4]         = {2, 2, 3, 4};
	index_t shape2[4]         = {2, 2, 3, 4};
	index_t shape3[4]         = {2, 2, 1, 4};
	index_t shape4[4]         = {1, 2, 1, 4};
	index_t shape5[3]         = {2, 3, 4};

	Tensor a                  = pascal_tensor_new_repeat(repeated_value, shape1, 4);
	Tensor b                  = pascal_tensor_new_repeat(repeated_value, shape2, 4);
	Tensor c                  = pascal_tensor_new_repeat(repeated_value, shape3, 4);
	Tensor d                  = pascal_tensor_new_repeat(repeated_value, shape4, 4);
	Tensor e                  = pascal_tensor_new_repeat(repeated_value, shape5, 3);

	BroadcastOutput b_output  = pascal_tensor_broadcast(a, b);
	Tensor          out1      = b_output->tensor;

	BroadcastOutput b_output2 = pascal_tensor_broadcast(a, c);
	Tensor          out2      = b_output2->tensor;

	BroadcastOutput b_output3 = pascal_tensor_broadcast(a, d);
	Tensor          out3      = b_output3->tensor;

	BroadcastOutput b_output4 = pascal_tensor_broadcast(a, e);
	Tensor          out4      = b_output4->tensor;

	for (int i = 0; i < a->ndim; i++) {
		arbiter_assert(out1->shape[i] == shape1[i]);
		arbiter_assert(out2->shape[i] == shape1[i]);
		arbiter_assert(out3->shape[i] == shape1[i]);
		arbiter_assert(out4->shape[i] == shape1[i]);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(c);
	pascal_tensor_free(d);
	pascal_tensor_free(e);

	pascal_tensor_broadcast_output_free(b_output);
	pascal_tensor_broadcast_output_free(b_output2);
	pascal_tensor_broadcast_output_free(b_output3);
	pascal_tensor_broadcast_output_free(b_output4);

	pascal_tensor_free(out1);
	pascal_tensor_free(out2);
	pascal_tensor_free(out3);
	pascal_tensor_free(out4);
}

static double operation_broad_cast_and_operate(double a, double b) {
	return a + b;
}

static void test_pascal_tensor_broadcast_and_operate() {
	double  values1[6]        = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape1[3]         = {3, 1, 2};
	index_t ndim1             = 3;

	double  values2[2]        = {50.0, 60.0};
	index_t shape2[2]         = {2, 1};
	index_t ndim2             = 2;

	Tensor a                  = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b                  = pascal_tensor_new(values2, shape2, ndim2);

	Tensor c                  = pascal_tensor_broadcast_and_operate(a, b, operation_broad_cast_and_operate);

	index_t expected_shape[3] = {3, 2, 2};
	for (int i = 0; i < c->ndim; i++) {
		arbiter_assert(c->shape[i] == expected_shape[i]);
	}

	double expected_values[12] = {51.0, 52.0, 61.0, 62.0, 53.0, 54.0, 63.0, 64.0, 55.0, 56.0, 65.0, 66.0};
	for (int i = 0; i < c->size; i++) {
		arbiter_assert(fabs(c->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);

	pascal_tensor_free(b);
	pascal_tensor_free(c);
}

static void test_pascal_tensor_broadcast_linalg() {
	double  values1_dot[6]           = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape1_dot[3]            = {3, 2, 1};
	index_t ndim1_dot                = 3;

	double  values2_dot[6]           = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape2_dot[3]            = {3, 2, 1};
	index_t ndim2_dot                = 3;

	Tensor a_dot                     = pascal_tensor_new(values1_dot, shape1_dot, ndim1_dot);
	Tensor b_dot                     = pascal_tensor_new(values2_dot, shape2_dot, ndim2_dot);

	index_t         out_shape_dot[2] = {1, 1};
	BroadcastOutput b_output         = pascal_tensor_broadcast_linalg(a_dot, b_dot, out_shape_dot, 2);
	Tensor          c_dot            = b_output->tensor;

	index_t expected_ndim_dot        = 3;
	arbiter_assert(c_dot->ndim == expected_ndim_dot);

	index_t expected_shape_dot[3] = {3, 1, 1};
	for (int i = 0; i < expected_ndim_dot; i++) {
		arbiter_assert(c_dot->shape[i] == expected_shape_dot[i]);
	}

	double  values1_matmul[12]          = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape1_matmul[3]            = {3, 2, 2};
	index_t ndim1_matmul                = 3;

	double  values2_matmul[2]           = {50.0, 60.0};
	index_t shape2_matmul[2]            = {2, 1};
	index_t ndim2_matmul                = 2;

	Tensor a_matmul                     = pascal_tensor_new(values1_matmul, shape1_matmul, ndim1_matmul);
	Tensor b_matmul                     = pascal_tensor_new(values2_matmul, shape2_matmul, ndim2_matmul);

	index_t         out_shape_matmul[2] = {a_matmul->shape[a_matmul->ndim - 2], b_matmul->shape[b_matmul->ndim - 1]};
	BroadcastOutput b_output_matmul     = pascal_tensor_broadcast_linalg(a_matmul, b_matmul, out_shape_matmul, 2);
	Tensor          c_matmul            = b_output_matmul->tensor;

	index_t expected_ndim_matmul        = 3;
	arbiter_assert(c_matmul->ndim == expected_ndim_matmul);

	index_t expected_shape_matmul[3] = {3, 2, 1};
	for (int i = 0; i < expected_ndim_matmul; i++) {
		arbiter_assert(c_matmul->shape[i] == expected_shape_matmul[i]);
	}

	pascal_tensor_free(a_dot);
	pascal_tensor_free(b_dot);
	pascal_tensor_free(c_dot);

	pascal_tensor_free(a_matmul);
	pascal_tensor_free(b_matmul);
	pascal_tensor_free(c_matmul);
	pascal_tensor_broadcast_output_free(b_output);
	pascal_tensor_broadcast_output_free(b_output_matmul);
}
#define NUM_TESTS 3

int main() {
	void (*tests[NUM_TESTS])() = {
			test_pascal_tensor_broadcast,
			test_pascal_tensor_broadcast_and_operate,
			test_pascal_tensor_broadcast_linalg,
	};
	arbiter_run_tests(NUM_TESTS, "pascal_tensor_broadcast", tests);
}
