#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

Tensor pascal_tensor_conv2d(Tensor a, Tensor filter, index_t stride[]) {
	pascal_tensor_assert(a->ndim == 4, "Input tensor must have 4 dimensions\n");
	pascal_tensor_assert(filter->ndim == 4, "Filter tensor must have 4 dimensions\n");
	pascal_tensor_assert(a->shape[1] == filter->shape[1], "The number of channels in the input tensor must be equal to the number of channels in the filter tensor\n");

	Tensor tensor      = pascal_tensor_init();

	index_t out_ndim   = 4;

	index_t* out_shape = malloc(sizeof(index_t) * out_ndim);
	out_shape[0]       = filter->shape[0];
	out_shape[1]       = a->shape[0];

	for (int i = 2; i < out_ndim; i++) {
		pascal_tensor_assert(stride[i] > 0, "Stride must be greater than 0\n");
		out_shape[i] = (a->shape[i] - filter->shape[i]) / stride[i - 2] + 1;
	}

	index_t* out_stride         = pascal_tensor_utils_default_stride(out_shape, out_ndim);
	index_t  out_size           = pascal_tensor_utils_size_from_shape(out_shape, out_ndim);

	index_t* filter_temp_shape  = malloc(sizeof(index_t) * 3);
	filter_temp_shape[0]        = filter->shape[1];
	filter_temp_shape[1]        = filter->shape[2];
	filter_temp_shape[2]        = filter->shape[3];

	index_t  filter_temp_ndim   = 3;
	index_t  filter_temp_size   = pascal_tensor_utils_size_from_shape(filter_temp_shape, filter_temp_ndim);
	index_t* filter_temp_stride = pascal_tensor_utils_default_stride(filter_temp_shape, filter_temp_ndim);

	Tensor temp_filter          = pascal_tensor_init();
	temp_filter->ndim           = filter_temp_ndim;
	temp_filter->size           = filter_temp_size;
	temp_filter->shape          = filter_temp_shape;
	temp_filter->_stride        = filter_temp_stride;
	temp_filter->values         = filter->values;

	index_t* temp_stride        = malloc(sizeof(index_t) * 3);
	temp_stride[0]              = 0;
	temp_stride[1]              = stride[0];
	temp_stride[2]              = stride[1];

	double* values              = malloc(sizeof(double) * out_size);
	index_t outer_fixed_size    = out_size / out_shape[0];

	for (int i = 0; i < filter->shape[0]; i++) {
		temp_filter->values = filter->values + (i * filter_temp_size);

		Tensor _t           = pascal_tensor_convolution(a, temp_filter, temp_stride);
		for (int j = 0; j < outer_fixed_size; j++) {
			values[i * outer_fixed_size + j] = _t->values[j];
		}

		// free(_t->shape);
		// free(_t->_stride);
		// free(_t);
		pascal_tensor_free(_t);
	}

	tensor->values  = values;
	tensor->ndim    = out_ndim;
	tensor->size    = out_size;
	tensor->shape   = out_shape;
	tensor->_stride = out_stride;

	Tensor out      = pascal_tensor_transpose(tensor, (index_t[]){1, 0, 2, 3});

	pascal_tensor_utils_unravel_and_replace(out);

	free(temp_filter->shape);
	free(temp_filter->_stride);
	free(temp_filter);
	free(temp_stride);

	pascal_tensor_free(tensor);
	return out;
}
