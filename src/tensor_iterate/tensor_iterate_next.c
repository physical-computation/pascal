#include "pascal.h"

double pascal_tensor_iterate_next(TensorIterator iterator, Tensor a) {
	pascal_tensor_iterate(iterator, a);
	return a->values[iterator->offset];
}
