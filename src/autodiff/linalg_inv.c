#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_linalg_inv(AutodiffNode a) {
	return pascal_autodiff_linalg_inv("linalg_inv", _autodiff_primitive_linalg_inv_forward, _autodiff_primitive_linalg_inv_gradient, a);
}
