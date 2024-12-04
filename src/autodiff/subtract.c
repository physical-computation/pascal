#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_subtract(AutodiffNode a, AutodiffNode b) {
	return _pascal_autodiff_operate("subtract", 2, _autodiff_primitive_subtract_forward, _autodiff_primitive_subtract_gradient, a, b);
}
