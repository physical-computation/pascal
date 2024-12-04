#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_square(AutodiffNode a) {
	return _pascal_autodiff_operate("square", 1, _autodiff_primitive_square_forward, _autodiff_primitive_square_gradient, a);
}
