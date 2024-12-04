#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_sigmoid(AutodiffNode a) {
	return _pascal_autodiff_operate("sigmoid", 1, _autodiff_primitive_sigmoid_forward, _autodiff_primitive_sigmoid_gradient, a);
}
