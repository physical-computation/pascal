#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_USE_UXHW
	#include "uxhw.h"

Tensor pascal_tensor_uncertain_normal(double mean[], double stddevs[], index_t shape[], index_t ndim) {
	Tensor tensor   = pascal_tensor_init();

	index_t size    = pascal_tensor_utils_size_from_shape(shape, ndim);

	index_t* _shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		_shape[i] = shape[i];
	}

	index_t* _stride = pascal_tensor_utils_default_stride(shape, ndim);
	double*  _values = malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		_values[i] = UxHwDoubleGaussDist(mean[i], stddevs[i]);
	}

	tensor->size    = size;
	tensor->ndim    = ndim;
	tensor->shape   = _shape;
	tensor->_stride = _stride;
	tensor->values  = _values;

	return tensor;
}
#else
	#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
		#include <gsl/gsl_randist.h>
		#include <gsl/gsl_rng.h>

static double _gaussian(double mean, double stddev, gsl_rng* r) {
	double unshifted = gsl_ran_gaussian_ziggurat(r, stddev);

	return mean + unshifted;
}

Tensor pascal_tensor_uncertain_normal(double mean[], double stddevs[], index_t shape[], index_t ndim, gsl_rng* r) {
	Tensor tensor   = pascal_tensor_init();

	index_t size    = pascal_tensor_utils_size_from_shape(shape, ndim);

	index_t* _shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		_shape[i] = shape[i];
	}

	index_t* _stride = pascal_tensor_utils_default_stride(shape, ndim);
	double*  _values = malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		_values[i] = _gaussian(mean[i], stddevs[i], r);
	}

	tensor->size    = size;
	tensor->ndim    = ndim;
	tensor->shape   = _shape;
	tensor->_stride = _stride;
	tensor->values  = _values;

	return tensor;
}
	#else

Tensor pascal_tensor_uncertain_normal(double mean[], double stddevs[], index_t shape[], index_t ndim) {
	Tensor tensor   = pascal_tensor_init();

	index_t size    = pascal_tensor_utils_size_from_shape(shape, ndim);

	index_t* _shape = malloc(sizeof(index_t) * ndim);
	for (int i = 0; i < ndim; i++) {
		_shape[i] = shape[i];
	}

	index_t* _stride = pascal_tensor_utils_default_stride(shape, ndim);
	double*  _values = malloc(sizeof(double) * size);

	for (int i = 0; i < size; i++) {
		_values[i] = pascal_tensor_random_sample_normal(mean[i], stddevs[i]);
	}

	tensor->size    = size;
	tensor->ndim    = ndim;
	tensor->shape   = _shape;
	tensor->_stride = _stride;
	tensor->values  = _values;

	return tensor;
}
	#endif
#endif
