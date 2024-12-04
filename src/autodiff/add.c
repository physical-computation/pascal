#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_add(AutodiffNode a, AutodiffNode b) {
	return _pascal_autodiff_operate("add", 2, _autodiff_primitive_add_forward, _autodiff_primitive_add_gradient, a, b);
}
