#ifndef AUTODIFF_PRIMITIVES_H
#define AUTODIFF_PRIMITIVES_H

#include "pascal_autodiff.h"

Tensor _autodiff_primitive_add_forward(Tensor* inputs);
Tensor _autodiff_primitive_add_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_subtract_forward(Tensor* inputs);
Tensor _autodiff_primitive_subtract_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_multiply_forward(Tensor* inputs);
Tensor _autodiff_primitive_multiply_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_reciprocal_forward(Tensor* inputs);
Tensor _autodiff_primitive_reciprocal_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_matmul_forward(Tensor* inputs);
Tensor _autodiff_primitive_matmul_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_exp_forward(Tensor* inputs);
Tensor _autodiff_primitive_exp_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_log_forward(Tensor* inputs);
Tensor _autodiff_primitive_log_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_square_forward(Tensor* inputs);
Tensor _autodiff_primitive_square_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_sum_all_forward(Tensor* inputs);
Tensor _autodiff_primitive_sum_all_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_prod_all_forward(Tensor* inputs);
Tensor _autodiff_primitive_prod_all_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_mean_all_forward(Tensor* inputs);
Tensor _autodiff_primitive_mean_all_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_sigmoid_forward(Tensor* inputs);
Tensor _autodiff_primitive_sigmoid_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_tanh_forward(Tensor* inputs);
Tensor _autodiff_primitive_tanh_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_relu_forward(Tensor* inputs);
Tensor _autodiff_primitive_relu_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_sin_forward(Tensor* inputs);
Tensor _autodiff_primitive_sin_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_linalg_inv_forward(Tensor* inputs);
Tensor _autodiff_primitive_linalg_inv_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

Tensor _autodiff_primitive_copy_forward(Tensor* inputs);
Tensor _autodiff_primitive_copy_gradient(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index);

#endif
