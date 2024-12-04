#include "pascal_autodiff.h"
#include "_pascal_autodiff_primitives.h"

AutodiffNode pascal_autodiff_matmul(AutodiffNode a, AutodiffNode b) {
	return _pascal_autodiff_operate("matmul", 2, _autodiff_primitive_matmul_forward, _autodiff_primitive_matmul_gradient, a, b);
}
