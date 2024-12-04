#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_prod_all(AutodiffNode a) {
	return _pascal_autodiff_operate("prod_all", 1, _autodiff_primitive_prod_all_forward, _autodiff_primitive_prod_all_gradient, a);
}
