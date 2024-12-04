#include "pascal_autodiff.h"
#include "pascal.h"

Tensor pascal_tensor_self_derivative(Tensor a) {
	return pascal_tensor_ones(a->shape, a->ndim);
}
