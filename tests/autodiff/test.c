#include "arbiter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal_autodiff.h"

static void test_pascal_tensor_self_derivative() {
	index_t shape[3] = {2, 2, 3};
	index_t ndim     = 3;

	Tensor a         = pascal_tensor_random_normal(0, 1, shape, ndim);

	Tensor self_grad = pascal_tensor_self_derivative(a);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(self_grad, (index_t[]){i, j, k}) - 1.0) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(self_grad);
}

static double gradient_operation(double x) {
	return 2 * x;
}

static void test_pascal_tensor_self_derivative_with_operation() {
	index_t shape[3]    = {2, 2, 3};
	index_t ndim        = 3;

	Tensor a            = pascal_tensor_random_normal(0, 1, shape, ndim);

	Tensor current_grad = pascal_tensor_ones(shape, ndim);

	Tensor self_grad    = pascal_tensor_self_derivative_with_operation(a, current_grad, gradient_operation);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(self_grad, (index_t[]){i, j, k}) - gradient_operation(pascal_tensor_get(a, (index_t[]){i, j, k}))) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(self_grad);
}

static void test_pascal_autodiff_init() {
	AutodiffNode node = pascal_autodiff_init();

	arbiter_assert(node->num_inputs == 0);
	arbiter_assert(node->is_parameter == false);
	arbiter_assert(node->_is_necessary_for_gradient == false);
	arbiter_assert(node->forward_fn == NULL);
	arbiter_assert(node->gradient_fn == NULL);
	arbiter_assert(node->chain_rule_fn == NULL);
	arbiter_assert(node->next == NULL);

	arbiter_assert(node->grad == NULL);
	arbiter_assert(node->forward == NULL);

	arbiter_assert(node->_transform_info->map_forward == NULL);
	arbiter_assert(node->_transform_info->map_gradient == NULL);
	arbiter_assert(node->_transform_info->map == false);
	arbiter_assert(fabs(node->_transform_info->clamp_min - 0) < ARBITER_FLOATINGPOINT_ACCURACY);
	arbiter_assert(fabs(node->_transform_info->clamp_max - 0) < ARBITER_FLOATINGPOINT_ACCURACY);
	arbiter_assert(node->_transform_info->clamp == false);

	pascal_autodiff_free(node);
}

static void test_pascal_autodiff_new() {
	Tensor       a_tensor = pascal_tensor_zeros((index_t[]){2, 2, 3}, 3);
	AutodiffNode node     = pascal_autodiff_new(a_tensor);

	arbiter_assert(node->num_inputs == 0);
	arbiter_assert(node->is_parameter == false);
	arbiter_assert(node->_is_necessary_for_gradient == false);
	arbiter_assert(node->forward_fn == NULL);
	arbiter_assert(node->gradient_fn == NULL);
	arbiter_assert(node->next == NULL);

	arbiter_assert(node->grad == NULL);
	arbiter_assert(node->forward == a_tensor);

	arbiter_assert(node->_transform_info->map_forward == NULL);
	arbiter_assert(node->_transform_info->map_gradient == NULL);
	arbiter_assert(node->_transform_info->map == false);
	arbiter_assert(fabs(node->_transform_info->clamp_min - 0) < ARBITER_FLOATINGPOINT_ACCURACY);
	arbiter_assert(fabs(node->_transform_info->clamp_max - 0) < ARBITER_FLOATINGPOINT_ACCURACY);
	arbiter_assert(node->_transform_info->clamp == false);

	pascal_autodiff_free(node);
}

static void test_pascal_autodiff_parameter() {
	Tensor       a_tensor = pascal_tensor_zeros((index_t[]){2, 2, 3}, 3);
	AutodiffNode node     = pascal_autodiff_parameter(a_tensor);

	arbiter_assert(node->num_inputs == 0);
	arbiter_assert(node->is_parameter == true);
	arbiter_assert(node->_is_necessary_for_gradient == true);
	arbiter_assert(node->forward_fn == NULL);
	arbiter_assert(node->gradient_fn == NULL);
	arbiter_assert(node->next == NULL);

	arbiter_assert(node->grad == NULL);
	arbiter_assert(node->forward == a_tensor);

	arbiter_assert(node->_transform_info->map_forward == NULL);
	arbiter_assert(node->_transform_info->map_gradient == NULL);
	arbiter_assert(node->_transform_info->map == false);
	arbiter_assert(fabs(node->_transform_info->clamp_min - 0) < ARBITER_FLOATINGPOINT_ACCURACY);
	arbiter_assert(fabs(node->_transform_info->clamp_max - 0) < ARBITER_FLOATINGPOINT_ACCURACY);
	arbiter_assert(node->_transform_info->clamp == false);

	pascal_autodiff_free(node);
}

static void test_computation_graph_creation() {
	double  repeated_value1 = 1.0;
	double  repeated_value2 = 2.3;
	index_t shape[3]        = {2, 2, 3};
	index_t ndim            = 3;

	Tensor a_tensor         = pascal_tensor_new_repeat(repeated_value1, shape, ndim);
	Tensor b_tensor         = pascal_tensor_new_repeat(repeated_value2, shape, ndim);

	AutodiffNode a          = pascal_autodiff_new(a_tensor);
	AutodiffNode b          = pascal_autodiff_new(b_tensor);

	AutodiffNode m1         = pascal_autodiff_add(a, b);

	arbiter_assert(m1->next[0] == a);
	arbiter_assert(m1->next[1] == b);
	arbiter_assert(m1->next[0]->next == NULL);
	arbiter_assert(m1->next[1]->next == NULL);

	pascal_autodiff_free(m1);
}

static void test_forward() {
	double  repeated_value1 = 1.0;
	double  repeated_value2 = 2.3;
	index_t shape[3]        = {2, 2, 3};
	index_t ndim            = 3;

	Tensor a_tensor         = pascal_tensor_new_repeat(repeated_value1, shape, ndim);
	Tensor b_tensor         = pascal_tensor_new_repeat(repeated_value2, shape, ndim);

	AutodiffNode a          = pascal_autodiff_new(a_tensor);
	AutodiffNode b          = pascal_autodiff_new(b_tensor);

	AutodiffNode m1         = pascal_autodiff_add(a, b);

	pascal_autodiff_compute_forward(m1);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(m1->forward, (index_t[]){i, j, k}) - 3.3) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_autodiff_free(m1);
}

static void test_updated_forward() {
	index_t ndim     = 3;
	index_t shape[3] = {2, 2, 3};
	Tensor  a_tensor = pascal_tensor_zeros(shape, ndim);
	Tensor  b_tensor = pascal_tensor_ones(shape, ndim);

	AutodiffNode a   = pascal_autodiff_new(a_tensor);
	AutodiffNode b   = pascal_autodiff_new(b_tensor);

	AutodiffNode m1  = pascal_autodiff_add(a, b);
	AutodiffNode m2  = pascal_autodiff_multiply(m1, a);

	pascal_autodiff_compute_forward(m2);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(m2->forward, (index_t[]){i, j, k}) - 0.0) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a_tensor);
	a->forward = pascal_tensor_new_repeat(2.0, shape, ndim);

	pascal_autodiff_compute_forward(m2);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(m2->forward, (index_t[]){i, j, k}) - 6.0) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_autodiff_free(m2);
}

static void test_gradient_of_root_and_leaf() {
	double  repeated_value1 = 1.0;
	double  repeated_value2 = 2.3;
	double  repeated_value3 = 3.4;
	index_t shape[3]        = {2, 2, 3};
	index_t ndim            = 3;

	Tensor a_tensor         = pascal_tensor_new_repeat(repeated_value1, shape, ndim);
	Tensor b_tensor         = pascal_tensor_new_repeat(repeated_value2, shape, ndim);
	Tensor c_tensor         = pascal_tensor_new_repeat(repeated_value3, shape, ndim);

	AutodiffNode a          = pascal_autodiff_parameter(a_tensor);
	AutodiffNode b          = pascal_autodiff_new(b_tensor);
	AutodiffNode c          = pascal_autodiff_new(c_tensor);

	AutodiffNode m1         = pascal_autodiff_add(a, b);
	AutodiffNode m2         = pascal_autodiff_multiply(m1, c);

	pascal_autodiff_compute_forward(m2);
	pascal_autodiff_compute_backward(m2);

	Tensor gradient = a->grad;

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(gradient, (index_t[]){i, j, k}) - 3.4) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_autodiff_free(m2);
}

static double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

static double sigmoid_grad(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

static void test_pascal_autodiff_map_operation() {
	double  values[12]    = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape[3]      = {3, 2, 2};
	index_t ndim          = 3;

	Tensor       a_tensor = pascal_tensor_new(values, shape, ndim);
	AutodiffNode a        = pascal_autodiff_parameter(a_tensor);

	AutodiffNode m        = pascal_autodiff_map(a, sigmoid, sigmoid_grad);

	pascal_autodiff_compute_forward(m);
	pascal_autodiff_compute_backward(m);

	Tensor gradient       = a->grad;

	index_t expected_ndim = 3;
	arbiter_assert(gradient->ndim == expected_ndim);

	index_t expected_shape[3] = {3, 2, 2};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(gradient->shape[i] == expected_shape[i]);
	}

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(gradient, (index_t[]){i, j, k}) - sigmoid_grad(pascal_tensor_get(a_tensor, (index_t[]){i, j, k}))) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_autodiff_free(m);
}

static double clamp_operation_gradient(double x, double clamp_min, double clamp_max) {
	if (x < clamp_min || x > clamp_max) {
		return 0;
	} else {
		return 1;
	}
}

static void test_pascal_autodiff_clamp_operation() {
	index_t ndim          = 3;
	index_t shape[3]      = {3, 2, 2};
	double  values[12]    = {0.03992382, -4.45962422, -0.42867344, 1.76058905, -1.31076798, -1.29633849, -1.8428066, 2.91669712, 4.10416207, -4.38678569, 0.76771607, -2.7582438};

	Tensor       a_tensor = pascal_tensor_new(values, shape, ndim);
	AutodiffNode a        = pascal_autodiff_parameter(a_tensor);

	double clamp_min      = -3.0;
	double clamp_max      = 3.0;

	AutodiffNode m        = pascal_autodiff_clamp(a, clamp_min, clamp_max);

	pascal_autodiff_compute_forward(m);
	pascal_autodiff_compute_backward(m);

	Tensor gradient = a->grad;

	arbiter_assert(m->forward->ndim == 3);
	arbiter_assert(m->forward->size == 12);
	for (int i = 0; i < m->forward->ndim; i++) {
		arbiter_assert(m->forward->shape[i] == shape[i]);
		arbiter_assert(m->forward->_stride[i] == a_tensor->_stride[i]);
	}

	double expected_values[12] = {0.03992382, -3.0, -0.42867344, 1.76058905, -1.31076798, -1.29633849, -1.8428066, 2.91669712, 3.0, -3.0, 0.76771607, -2.7582438};
	for (int i = 0; i < m->forward->size; i++) {
		arbiter_assert(fabs(m->forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	index_t expected_ndim = 3;
	arbiter_assert(gradient->ndim == expected_ndim);

	index_t expected_shape[3] = {3, 2, 2};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(gradient->shape[i] == expected_shape[i]);
	}

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			for (int k = 0; k < shape[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(gradient, (index_t[]){i, j, k}) - clamp_operation_gradient(pascal_tensor_get(a_tensor, (index_t[]){i, j, k}), clamp_min, clamp_max)) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_autodiff_free(m);
}

static void test_is_necessary_for_gradient_propagation() {
	Tensor a_tensor = pascal_tensor_ones((index_t[]){2, 2}, 2);
	Tensor b_tensor = pascal_tensor_ones((index_t[]){2, 2}, 2);
	Tensor c_tensor = pascal_tensor_ones((index_t[]){2, 2}, 2);

	AutodiffNode a  = pascal_autodiff_parameter(a_tensor);
	AutodiffNode b  = pascal_autodiff_parameter(b_tensor);
	AutodiffNode c  = pascal_autodiff_new(c_tensor);

	AutodiffNode d  = pascal_autodiff_add(a, b);
	AutodiffNode e  = pascal_autodiff_add(d, c);
	AutodiffNode f  = pascal_autodiff_add(c, c);
	AutodiffNode g  = pascal_autodiff_add(e, f);

	arbiter_assert(d->_is_necessary_for_gradient == true);
	arbiter_assert(e->_is_necessary_for_gradient == true);
	arbiter_assert(f->_is_necessary_for_gradient == false);
	arbiter_assert(g->_is_necessary_for_gradient == true);

	pascal_autodiff_free(g);
}

static void test_chain_rule_non_zeros() {
	index_t      ndim_a     = 4;
	index_t      shape_a[4] = {2, 2, 100, 2};
	AutodiffNode a          = pascal_autodiff_random_normal_parameter(0, 1, shape_a, ndim_a);

	index_t      ndim_b     = 2;
	index_t      shape_b[2] = {100, 2};
	AutodiffNode b          = pascal_autodiff_random_normal(0, 1, shape_b, ndim_b);

	AutodiffNode c          = pascal_autodiff_add(a, b);

	for (int i = 0; i < 2; i++) {
		c = pascal_autodiff_add(c, b);
	}

	pascal_autodiff_compute_forward(c);
	pascal_autodiff_compute_backward(c);

	pascal_autodiff_free(c);
}

static void test_add_add_chain_rule() {
	AutodiffNode a = pascal_autodiff_random_uniform_parameter(-1, 1, (index_t[]){2, 3, 4}, 3);
	AutodiffNode b = pascal_autodiff_random_uniform(-1, 1, (index_t[]){2, 3, 4}, 3);
	AutodiffNode c = pascal_autodiff_random_uniform(-1, 1, (index_t[]){2, 3, 4}, 3);

	AutodiffNode d = pascal_autodiff_add(a, b);
	d              = pascal_autodiff_add(d, c);

	pascal_autodiff_compute_forward(d);
	pascal_autodiff_compute_backward(d);

	index_t expected_ndim     = 3;
	index_t expected_size     = 24;
	index_t expected_shape[3] = {2, 3, 4};

	arbiter_assert(a->grad->ndim == expected_ndim);
	arbiter_assert(a->grad->size == expected_size);
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(a->grad->shape[i] == expected_shape[i]);
	}

	pascal_autodiff_free(d);
}

static Tensor pascal_tensor_squeeze(Tensor a) {
	index_t  ndim  = a->ndim - 1;
	index_t* shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		shape[i] = a->shape[i + 1];
	}

	Tensor t = pascal_tensor_new(a->values, shape, ndim);

	free(shape);
	return t;
}

static void update_weights(AutodiffNode weight, double learning_rate) {
	Tensor weighted_gradient = pascal_tensor_scalar_multiply(weight->grad, learning_rate);
	Tensor new_weights       = pascal_tensor_subtract(weight->forward, weighted_gradient);

	pascal_tensor_free(weight->forward);
	pascal_tensor_free(weighted_gradient);

	weight->forward = new_weights;
}

static void test_loss_during_neural_network_training() {
	index_t num_data_points          = 10;
	index_t x_dim                    = 3;
	index_t y_dim                    = 1;

	index_t n_nodes                  = 10;

	index_t ndim                     = 2;
	index_t x_shape[2]               = {10, 3};
	double  x_values[30]             = {-1.85830188, 6.49783171, -5.28644911, -8.64860012, 7.26167869, -6.80747631, 3.05319221, 2.13969528, 6.08561453, -7.7652272, 5.04327755, -7.63837908, -6.58659673, 1.69170154, -3.86355285, -2.94409609, -5.44211817, -7.32981358, -9.10707192, 8.81635902, 5.04970648, 2.49813017, 5.37001898, -8.07558233, -3.13279303, -9.19288847, -9.32337246, 7.48993695, 5.70166202, -9.41504528};

	Tensor       x_data              = pascal_tensor_new(x_values, x_shape, ndim);
	AutodiffNode x                   = pascal_autodiff_new(x_data);

	index_t y_shape[2]               = {10, 1};
	double  y_values[10]             = {1.04437514, 0.97880481, 0.06629005, 1.05698838, 0.71818224, 0.01391952, -0.04546158, 1.09317289, 0.0722612, 0.96746236};

	Tensor       y_data              = pascal_tensor_new(y_values, y_shape, ndim);
	AutodiffNode y                   = pascal_autodiff_new(y_data);

	index_t t0_ndim                  = 2;
	index_t t0_shape[2]              = {3, 10};
	double  t0_values[30]            = {3.285286, -0.723522, 0.217484, -1.068676, 0.806425, -1.374203, 0.594250, -2.378225, 0.891999, 1.264324, -0.511385, 0.932220, -0.569741, -1.240200, -0.399840, 0.033192, 1.563455, -0.369499, 0.038533, -0.088953, 1.662209, 0.016993, 0.830817, 0.717879, -1.485082, -0.722157, -0.264169, 0.716676, 0.960650, 0.405095};

	Tensor       t0                  = pascal_tensor_new(t0_values, t0_shape, t0_ndim);
	AutodiffNode w0                  = pascal_autodiff_parameter(t0);

	index_t t1_ndim                  = 2;
	index_t t1_shape[2]              = {10, 1};
	double  t1_values[10]            = {-0.447488, -0.483253, -0.163882, 1.369771, 0.293023, -0.878082, 1.372940, -1.099247, 0.331679, 0.093458};

	Tensor       t1                  = pascal_tensor_new(t1_values, t1_shape, t1_ndim);
	AutodiffNode w1                  = pascal_autodiff_parameter(t1);

	AutodiffNode x1                  = pascal_autodiff_matmul(x, w0);
	AutodiffNode s1                  = pascal_autodiff_sigmoid(x1);

	AutodiffNode x2                  = pascal_autodiff_matmul(s1, w1);
	AutodiffNode y_out               = pascal_autodiff_sigmoid(x2);

	AutodiffNode loss_diff           = pascal_autodiff_subtract(y_out, y);
	AutodiffNode loss_square         = pascal_autodiff_square(loss_diff);

	AutodiffNode loss                = pascal_autodiff_mean_all(loss_square);

	double learning_rate             = 2.0;

	index_t n_iterations             = 10;
	double  expected_loss_values[20] = {0.2801700728791022, 0.21691120827086108, 0.17680471935072567, 0.14704341632032297, 0.13082086754685052, 0.11422570327600139, 0.09637724264798639, 0.08128663547847911, 0.07153757939591054, 0.06439117412955915, 0.05835790666494478, 0.05397108582669565, 0.05066569839866375, 0.047930519377459735, 0.04556637638679963, 0.04347837535927217, 0.04161000630602451, 0.039922694583625955, 0.03838797953809792, 0.03698387587567044};

	for (int i = 0; i < n_iterations; i++) {
		pascal_autodiff_compute_forward(loss);
		arbiter_assert(fabs(pascal_tensor_get(loss->forward, (index_t[]){0}) - expected_loss_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);

		pascal_autodiff_compute_backward(loss);

		update_weights(w0, learning_rate);
		update_weights(w1, learning_rate);
	}

	pascal_autodiff_free(loss);
}

#define NUM_TESTS 15

int main() {
	void (*tests[NUM_TESTS])() = {
			test_pascal_tensor_self_derivative,
			test_pascal_tensor_self_derivative_with_operation,
			test_pascal_autodiff_init,
			test_pascal_autodiff_new,
			test_pascal_autodiff_parameter,
			test_computation_graph_creation,
			test_forward,
			test_updated_forward,
			test_gradient_of_root_and_leaf,
			test_pascal_autodiff_map_operation,
			test_pascal_autodiff_clamp_operation,
			test_is_necessary_for_gradient_propagation,
			test_chain_rule_non_zeros,
			test_add_add_chain_rule,
			test_loss_during_neural_network_training,
	};

	arbiter_run_tests(NUM_TESTS, "autodiff", tests);
}
