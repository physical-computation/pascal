#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_sum_all(AutodiffNode a) {
	return _pascal_autodiff_operate("sum_all", 1, _autodiff_primitive_sum_all_forward, _autodiff_primitive_sum_all_gradient, a);
}
