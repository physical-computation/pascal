#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_log(AutodiffNode a) {
	return _pascal_autodiff_operate("log", 1, _autodiff_primitive_log_forward, _autodiff_primitive_log_gradient, a);
}
