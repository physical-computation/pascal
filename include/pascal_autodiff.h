#ifndef autodiff_H
#define autodiff_H

#include "pascal.h"

typedef enum {
	AutodiffNodeOperationSin = 0,
	AutodiffNodeOperationTanh,
} AutodiffNodeOperation;

typedef struct transform_info_def {
	double (*map_forward)(double x);
	double (*map_gradient)(double x);
	bool   map;
	double clamp_min;
	double clamp_max;
	bool   clamp;
} TransformInfo_D, *TransformInfo;

typedef struct gradient_info_def {
	bool     is_eye;
	index_t  ndim;
	index_t* left;
	index_t* right;
} GradientInfo_D, *GradientInfo;

typedef struct pascal_autodiff_def {
	char* operation;
	int   num_inputs;
	bool  is_parameter;
	bool  _is_necessary_for_gradient;
	Tensor (*forward_fn)(Tensor* inputs);
	Tensor (*gradient_fn)(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);
	Tensor (*chain_rule_fn)(Tensor current_grad, Tensor gradient, index_t index);
	struct pascal_autodiff_def** next;
	Tensor                       grad;
	Tensor                       forward;
	TransformInfo                _transform_info;
} pascal_autodiff_D, *AutodiffNode;

Tensor       pascal_tensor_self_derivative(Tensor a);
Tensor       pascal_tensor_self_derivative_with_operation(Tensor a, Tensor current_grad, double (*gradient)(double));

void         pascal_autodiff_print(AutodiffNode node);

AutodiffNode pascal_autodiff_init();

AutodiffNode pascal_autodiff_new(Tensor value);
AutodiffNode pascal_autodiff_parameter(Tensor value);
AutodiffNode pascal_autodiff_constant_scalar(double value);

AutodiffNode pascal_autodiff_random_uniform(double min, double max, index_t shape[], index_t ndim);
AutodiffNode pascal_autodiff_random_normal(double mean, double variance, index_t shape[], index_t ndim);
AutodiffNode pascal_autodiff_random_uniform_parameter(double min, double max, index_t shape[], index_t ndim);
AutodiffNode pascal_autodiff_random_normal_parameter(double mean, double variance, index_t shape[], index_t ndim);

void         pascal_autodiff_free(AutodiffNode node);

AutodiffNode _pascal_autodiff_operate(char operation[], size_t num_inputs, Tensor (*forward)(Tensor* inputs), Tensor (*gradient)(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index), ...);
AutodiffNode pascal_autodiff_operate(AutodiffNodeOperation operation, ...);

void         pascal_autodiff_compute_forward(AutodiffNode node);
void         pascal_autodiff_compute_backward(AutodiffNode node);

AutodiffNode pascal_autodiff_map(AutodiffNode, double (*map_forward)(double), double (*map_gradient)(double));

AutodiffNode pascal_autodiff_clamp(AutodiffNode, double min, double max);

AutodiffNode pascal_autodiff_add(AutodiffNode a, AutodiffNode b);
AutodiffNode pascal_autodiff_subtract(AutodiffNode a, AutodiffNode b);
AutodiffNode pascal_autodiff_multiply(AutodiffNode a, AutodiffNode b);
AutodiffNode pascal_autodiff_reciprocal(AutodiffNode a);

AutodiffNode pascal_autodiff_matmul(AutodiffNode a, AutodiffNode b);
// AutodiffNode pascal_autodiff_linalg_inv(AutodiffNode a);

AutodiffNode pascal_autodiff_exp(AutodiffNode a);
AutodiffNode pascal_autodiff_log(AutodiffNode a);
AutodiffNode pascal_autodiff_square(AutodiffNode a);

AutodiffNode pascal_autodiff_sum_all(AutodiffNode a);
AutodiffNode pascal_autodiff_prod_all(AutodiffNode a);
AutodiffNode pascal_autodiff_mean_all(AutodiffNode a);

AutodiffNode pascal_autodiff_sigmoid(AutodiffNode a);
AutodiffNode pascal_autodiff_tanh(AutodiffNode a);
AutodiffNode pascal_autodiff_relu(AutodiffNode a);
AutodiffNode pascal_autodiff_sin(AutodiffNode a);

AutodiffNode pascal_autodiff_copy(AutodiffNode a);

#endif
