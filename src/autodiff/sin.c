#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_sin(AutodiffNode a) {
	return _pascal_autodiff_operate("sin", 1, _autodiff_primitive_sin_forward, _autodiff_primitive_sin_gradient, a);
}
