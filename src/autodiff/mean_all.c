#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_mean_all(AutodiffNode a) {
	return _pascal_autodiff_operate("mean_all", 1, _autodiff_primitive_mean_all_forward, _autodiff_primitive_mean_all_gradient, a);
}
