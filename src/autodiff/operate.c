#include "pascal_autodiff.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

AutodiffNode _pascal_autodiff_operate(char operation[], size_t num_inputs, Tensor (*forward)(Tensor* inputs), Tensor (*gradient)(Tensor* inputs, Tensor forward, Tensor current_grad, index_t index), ...) {
	va_list args;
	va_start(args, gradient);

	AutodiffNode new_node = pascal_autodiff_init();

	new_node->operation   = operation;

	new_node->num_inputs  = num_inputs;
	new_node->forward_fn  = forward;
	new_node->gradient_fn = gradient;

	Tensor* inputs        = malloc(num_inputs * sizeof(Tensor));

	new_node->next        = malloc(num_inputs * sizeof(AutodiffNode));

	for (int i = 0; i < num_inputs; i++) {
		AutodiffNode node                    = va_arg(args, AutodiffNode);
		(new_node->next)[i]                  = node;
		inputs[i]                            = node->forward;
		new_node->_is_necessary_for_gradient = new_node->_is_necessary_for_gradient || node->_is_necessary_for_gradient;
	}

	va_end(args);

	free(inputs);

	return new_node;
}

AutodiffNode pascal_autodiff_operate(AutodiffNodeOperation operation, ...) {
	va_list args;
	va_start(args, operation);

	AutodiffNode output;
	switch (operation) {
		case AutodiffNodeOperationSin: {
			AutodiffNode input = va_arg(args, AutodiffNode);
			output             = pascal_autodiff_sin(input);
		} break;
		case AutodiffNodeOperationTanh: {
			AutodiffNode input = va_arg(args, AutodiffNode);
			output             = pascal_autodiff_tanh(input);
		} break;
		default:
			exit(1);
	}

	va_end(args);
	return output;
}
