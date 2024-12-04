#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_tanh(AutodiffNode a) {
	return _pascal_autodiff_operate("tanh", 1, _autodiff_primitive_tanh_forward, _autodiff_primitive_tanh_gradient, a);
}
