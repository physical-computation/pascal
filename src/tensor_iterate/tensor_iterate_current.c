#include "pascal.h"

double pascal_tensor_iterate_current(TensorIterator iterator, Tensor a) {
	return a->values[iterator->offset];
}
