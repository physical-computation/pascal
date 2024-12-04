#ifndef BNN_SIMPLE_EVAL_BNN_H
#define BNN_SIMPLE_EVAL_BNN_H

#include <stdio.h>
#include <stdlib.h>

#include "pascal_autodiff.h"
#include "pascal.h"

Tensor bnn_pascal_tensor_squeeze(Tensor a);

AutodiffNode bnn_log_gaussian(AutodiffNode w, AutodiffNode mean, AutodiffNode std);
AutodiffNode bnn_log_gaussian_mean(AutodiffNode w, AutodiffNode mean);

AutodiffNode* bnn_init_weights(char type[], index_t sizes[], index_t input_dim, index_t output_dim, index_t n_layers);
AutodiffNode* bnn_init_epsilons(AutodiffNode w_means[], index_t n_layers);

AutodiffNode* bnn_calculate_weights(AutodiffNode w_means[], AutodiffNode w_stds[], AutodiffNode epsilons[], index_t n_layers);
AutodiffNode* bnn_calculate_stds(AutodiffNode w_rhos[], AutodiffNode epsilons[], index_t n_layers);

AutodiffNode bnn_mse(AutodiffNode y, AutodiffNode y_out, AutodiffNode w, AutodiffNode w_mean, AutodiffNode w_std);

void bnn_sample_epsilons(AutodiffNode epsilons[], index_t n_layers);
void bnn_update_weights(AutodiffNode weights, double learning_rate);

void bnn_save_tensor(FILE* location, Tensor tensor);
void bnn_save_loss_values(char* location, double loss_values[], index_t n_values);
#endif
