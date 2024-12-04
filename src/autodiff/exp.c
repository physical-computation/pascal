#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_exp(AutodiffNode a) {
	return _pascal_autodiff_operate("exp", 1, _autodiff_primitive_exp_forward, _autodiff_primitive_exp_gradient, a);
}
