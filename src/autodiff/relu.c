#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_relu(AutodiffNode a) {
	return _pascal_autodiff_operate("relu", 1, _autodiff_primitive_relu_forward, _autodiff_primitive_relu_gradient, a);
}
