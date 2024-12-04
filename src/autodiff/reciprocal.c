#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_reciprocal(AutodiffNode a) {
	return _pascal_autodiff_operate("reciprocal", 1, _autodiff_primitive_reciprocal_forward, _autodiff_primitive_reciprocal_gradient, a);
}
