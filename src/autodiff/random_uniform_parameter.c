#include "pascal_autodiff.h"
#include "pascal.h"

AutodiffNode pascal_autodiff_random_uniform_parameter(double min, double max, index_t shape[], index_t ndim) {
	Tensor t = pascal_tensor_random_uniform(min, max, shape, ndim);

	return pascal_autodiff_parameter(t);
}
