#include "pascal_autodiff.h"
#include "pascal.h"

AutodiffNode pascal_autodiff_random_normal(double mean, double variance, index_t shape[], index_t ndim) {
	Tensor t = pascal_tensor_random_normal(mean, variance, shape, ndim);

	return pascal_autodiff_new(t);
}
