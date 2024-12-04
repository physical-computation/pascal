#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_copy(AutodiffNode a) {
	return _pascal_autodiff_operate("copy", 1, _autodiff_primitive_copy_forward, _autodiff_primitive_copy_gradient, a);
}
