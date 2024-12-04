#include <stdio.h>
#include <stdlib.h>

#include "pascal.h"

#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
	#include <gsl/gsl_blas.h>
	#include <gsl/gsl_linalg.h>
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
	#include <cblas.h>
	#ifdef VALGRIND
		#include <lapacke.h>
	#else
		#ifdef DARWIN
			#include <clapack.h>
		#else
			#include <lapack.h>
		#endif
	#endif
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
	#include "clapack/f2c.h"
	#include "clapack/blaswrap.h"
	#include "clapack/clapack.h"
#endif

Tensor pascal_tensor_scalar_multiply(Tensor a, double k) {
	Tensor c = pascal_tensor_new(a->values, a->shape, a->ndim);
#if TENSOR_BACKEND == TENSOR_BACKEND_GSL
	gsl_vector_view v = gsl_vector_view_array(c->values, c->size);
	gsl_blas_dscal(k, &v.vector);
	c->values = v.vector.data;
#elif TENSOR_BACKEND == TENSOR_BACKEND_BLAS
	cblas_dscal(a->size, k, c->values, 1);
#elif TENSOR_BACKEND == TENSOR_BACKEND_CLAPACK
	integer ONE = 1;
	dscal_(&a->size, &k, c->values, &ONE);
#endif
	return c;
}
