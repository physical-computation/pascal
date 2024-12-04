#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

static double max_over_array(double* a, index_t size) {
	double max = a[0];
	for (int i = 1; i < size; i++) {
		if (a[i] > max) {
			max = a[i];
		}
	}
	return max;
}

Tensor pascal_tensor_max_pool(Tensor a, index_t filter_shape[], index_t stride[], index_t filter_ndim) {
	pascal_tensor_assert(a->ndim >= filter_ndim, "The filter must have less or equal dimensions than the input tensor\n");
	for (int i = 0; i < filter_ndim; i++) {
		pascal_tensor_assert(stride[i] > 0, "Stride must be greater than 0\n");
	}

	Tensor tensor      = pascal_tensor_init();

	index_t out_ndim   = a->ndim;

	index_t* out_shape = malloc(sizeof(index_t) * out_ndim);
	for (int i = 0; i < out_ndim; i++) {
		if (i < out_ndim - filter_ndim) {
			out_shape[i] = a->shape[i];
		} else {
			index_t filter_i = i - (out_ndim - filter_ndim);
			out_shape[i]     = (a->shape[i] - filter_shape[filter_i]) / stride[filter_i] + 1;
		}
	}

	index_t* out_stride  = pascal_tensor_utils_default_stride(out_shape, out_ndim);
	index_t  out_size    = pascal_tensor_utils_size_from_shape(out_shape, out_ndim);
	index_t  filter_size = pascal_tensor_utils_size_from_shape(filter_shape, filter_ndim);

	double* values       = malloc(sizeof(double) * out_size);

	index_t* indexes     = malloc(sizeof(index_t) * out_ndim);
	for (int i = 0; i < out_size; i++) {
		pascal_tensor_utils_index_from_linear_index(indexes, i, out_stride, out_ndim);

		index_t* start_indexes = pascal_tensor_utils_get_convolution_start_index(indexes, stride, out_ndim, filter_ndim);

		double* a_values       = pascal_tensor_utils_get_convolution_array(a, filter_shape, filter_size, filter_ndim, start_indexes);

		values[i]              = max_over_array(a_values, filter_size);

		free(start_indexes);
		free(a_values);
	}
	free(indexes);

	tensor->ndim    = out_ndim;
	tensor->size    = out_size;
	tensor->shape   = out_shape;
	tensor->_stride = out_stride;
	tensor->values  = values;

	return tensor;
}
