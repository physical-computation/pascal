#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
	#include <gsl/gsl_blas.h>
	#include <gsl/gsl_linalg.h>
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
	#include <cblas.h>
	#ifdef VALGRIND
		#include <lapacke.h>
	#else
		#ifdef DARWIN
			#include <clapack.h>
		#else
			#include <lapack.h>
		#endif
	#endif
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
	#include "clapack/f2c.h"
	#include "clapack/blaswrap.h"
	#include "clapack/clapack.h"
#endif

Tensor pascal_tensor_convolution(Tensor a, Tensor filter, index_t stride[]) {
	pascal_tensor_assert(a->ndim >= filter->ndim, "The filter must have less or equal dimensions than the input tensor\n");
	for (int i = 0; i < filter->ndim; i++) {
		if (stride[i] == 0) {
			pascal_tensor_assert(a->shape[a->ndim - filter->ndim + i] == filter->shape[i], "If stride is 0 for a dimension, then the filter must have the same shape as the input tensor at that dimension\n");
		}
	}

	Tensor tensor      = pascal_tensor_init();

	index_t out_ndim   = a->ndim;

	index_t* out_shape = malloc(sizeof(index_t) * out_ndim);
	for (int i = 0; i < out_ndim; i++) {
		if (i < out_ndim - filter->ndim) {
			out_shape[i] = a->shape[i];
		} else {
			index_t filter_i = i - (out_ndim - filter->ndim);
			if (stride[filter_i] == 0) {
				out_shape[i] = 1;
			} else {
				out_shape[i] = (a->shape[i] - filter->shape[filter_i]) / stride[filter_i] + 1;
			}
		}
	}

	index_t* out_stride      = pascal_tensor_utils_default_stride(out_shape, out_ndim);
	index_t  out_size        = pascal_tensor_utils_size_from_shape(out_shape, out_ndim);

	Tensor  flattened_filter = pascal_tensor_flatten(filter);
	double* filter_values    = flattened_filter->values;

	const index_t ORDER      = filter->size;

	double* values           = malloc(sizeof(double) * out_size);

	index_t* indexes         = malloc(sizeof(index_t) * out_ndim);
	for (int i = 0; i < out_size; i++) {
		pascal_tensor_utils_index_from_linear_index(indexes, i, out_stride, out_ndim);

		index_t* start_indexes = pascal_tensor_utils_get_convolution_start_index(indexes, stride, out_ndim, filter->ndim);

		double* a_values       = pascal_tensor_utils_get_convolution_array(a, filter->shape, filter->size, filter->ndim, start_indexes);

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
		double          result;
		gsl_vector_view a = gsl_vector_view_array(a_values, ORDER);
		gsl_vector_view b = gsl_vector_view_array(filter_values, ORDER);
		gsl_blas_ddot(&a.vector, &b.vector, &result);

		values[i] = result;
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
		values[i] = cblas_ddot(ORDER, a_values, 1, filter_values, 1);
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
		integer n    = ORDER;
		integer incx = 1;
		integer incy = 1;
		values[i]    = ddot_(&n, a_values, &incx, filter_values, &incy);
#endif
		free(start_indexes);
		free(a_values);
	}
	free(indexes);

	pascal_tensor_free(flattened_filter);

	tensor->ndim    = out_ndim;
	tensor->size    = out_size;
	tensor->shape   = out_shape;
	tensor->_stride = out_stride;
	tensor->values  = values;
	return tensor;
}
