#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_multiply(AutodiffNode a, AutodiffNode b) {
	return _pascal_autodiff_operate("multiply", 2, _autodiff_primitive_multiply_forward, _autodiff_primitive_multiply_gradient, a, b);
}
