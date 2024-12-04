#include "pascal_autodiff.h"
#include "pascal.h"

AutodiffNode pascal_autodiff_constant_scalar(double value) {
	return pascal_autodiff_new(pascal_tensor_new((double[]){value}, (index_t[]){1}, 1));
}
