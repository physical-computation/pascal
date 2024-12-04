#ifndef PASCAL_H
#define PASCAL_H

#ifndef TENSOR_BACKEND_GSL
	#define TENSOR_BACKEND_GSL 0
#endif

#ifndef TENSOR_BACKEND_BLAS
	#define TENSOR_BACKEND_BLAS 1
#endif

#ifndef TENSOR_BACKEND_CLAPACK
	#define TENSOR_BACKEND_CLAPACK 2
#endif

#ifndef TENSOR_BACKEND
	#define TENSOR_BACKEND TENSOR_BACKEND_CLAPACK
#endif

#ifndef TENSOR_USE_SIMD
	#define TENSOR_USE_SIMD 0
#endif

#ifndef TENSOR_PRINT_VERBOSE
	#define TENSOR_PRINT_VERBOSE 0
#endif

#ifndef TENSOR_USE_ASSERT
	#define TENSOR_USE_ASSERT 1
#endif

#include <stdbool.h>
#include <stdlib.h>

#if TENSOR_USE_ASSERT
	#define pascal_tensor_assert(condition, message) assert(condition &&message)
#else
	#define pascal_tensor_assert(condition, message)
#endif

#if TENSOR_USE_ASSERT
	#include <assert.h>
#endif

#ifndef TENSOR_USE_UXHW
	#define TENSOR_USE_UXHW 0
#endif

#ifndef M_PI
	#define M_PI 3.14159265358979323846264338327950288 /* pi             */
#endif
typedef unsigned int index_t;

typedef struct pascal_tensor_def {
	index_t  size;
	index_t  ndim;
	index_t* shape;
	index_t* _stride;
	index_t* _transpose_map;
	index_t* _transpose_map_inverse;
	double*  values;

} Tensor_D, *Tensor;

/**
 * @brief Print a tensor
 * \par
 * Print the tensor to the console in a human-readable format.
 * @param a The tensor to print
 */
void pascal_tensor_print(Tensor a);

/**
 * @brief Free a tensor
 * \par
 * Free the memory allocated for the tensor.
 * @param tensor The tensor to free
 */
void pascal_tensor_free(Tensor tensor);

/**
 * @brief Get the value of a tensor at a specific index
 * \par
 * Get the value of a tensor at a specific index. The index is specified by an array of indexes, one for each dimension.
 * Only the first a->ndim elements of the indexes array are used.
 * @param a The tensor to get the value from
 * @param indexes The indexes of the value to get. e.g., {0, 1, 2} would get the value at index (0, 1, 2) of a 3D tensor.
 * @return The value of the tensor at the specified index
 */
double pascal_tensor_get(Tensor a, index_t indexes[]);

/**
 * @brief Initialize an empty tensor object
 * \par
 * Initialize an empty tensor object. The tensor object must be freed with pascal_tensor_free.
 * @return The initialized tensor object
 * @see pascal_tensor_free
 * @see pascal_tensor_new
 * @see pascal_tensor_new_no_malloc
 */
Tensor pascal_tensor_init();

/**
 * @brief Create a new tensor
 * \par
 * Create a new tensor with the specified values, shape, and number of dimensions. The tensor object must be freed with pascal_tensor_free.
 * @note The values and shapes arrays will be copied into the tensor object.
 * @param values The values of the tensor
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The new tensor object
 * @see pascal_tensor_free
 * @see pascal_tensor_init
 * @see pascal_tensor_new_no_malloc
 */
Tensor pascal_tensor_new(double values[], index_t shape[], index_t ndim);

/**
 * @brief Create a new tensor without allocating memory for the values array.
 * \par
 * Create a new tensor without allocating memory for the values. The tensor object must be freed with pascal_tensor_free.
 * @note The values array will not be copied into the tensor object. Therefore, do not free the values, and they must be allocated before calling this function.
 * @param values The values of the tensor
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The new tensor object
 * @see pascal_tensor_free
 * @see pascal_tensor_init
 * @see pascal_tensor_new
 */
Tensor pascal_tensor_new_no_malloc(double values[], index_t shape[], index_t ndim);

/**
 * @brief Create a new tensor with a range of values
 * \par
 * Create a new tensor with a range of values from start to end, inclusive, with a specified number of values.
 * @param start The starting value of the range
 * @param end The ending value of the range
 * @param num The number of values in the range
 * @return The new tensor object
 */
Tensor pascal_tensor_linspace(double start, double end, index_t num);

/**
 * @brief Create a new identity matrix of size n.
 * \par
 * Create a new identity matrix of size n.
 * @param n The size of the identity matrix
 * @return The new tensor object
 */

Tensor pascal_tensor_eye(index_t n);

/**
 * @brief Create a new tensor with a repeated value
 * \par
 * Create a new tensor with a repeated value and the specified shape.
 * @param repeated_value The value to repeat
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The new tensor object
 */
Tensor pascal_tensor_new_repeat(double repeated_value, index_t shape[], index_t ndim);

/**
 * @brief Create a new tensor with zeros everywhere
 * \par
 * Create a new tensor with zeros and the specified shape.
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The new tensor object
 */
Tensor pascal_tensor_zeros(index_t shape[], index_t ndim);

/**
 * @brief Create a new tensor with ones everywhere
 * \par
 * Create a new tensor with ones and the specified shape.
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The new tensor object
 */
Tensor pascal_tensor_ones(index_t shape[], index_t ndim);

/**
 * @brief Create a new tensor that is a copy of the given tensor
 * \par
 * Create a new tensor that is a copy of the given tensor.
 * @param a The tensor to copy
 * @return The new tensor object
 */
Tensor pascal_tensor_copy(Tensor a);

/**
 * @brief Reshape a tensor
 * \par
 * Reshape a tensor to the specified shape.
 * @param a The tensor to reshape
 * @param new_shape The new shape of the tensor
 * @param ndim The number of dimensions of the new shape
 * @return The reshaped tensor
 */
Tensor pascal_tensor_reshape(Tensor a, index_t new_shape[], index_t ndim);

/**
 * @brief Transpose a tensor
 * \par Transpose a tensor according to the specified transpose map.
 * @param a The tensor to transpose
 * @param transpose_map The transpose map
 * @return The transposed tensor
 */
Tensor pascal_tensor_transpose(Tensor a, index_t transpose_map[]);

/**
 * @brief Tile a tensor
 * \par
 * Tile a tensor according to the specified tile map. Tiling creates a new tensor by repeating the tensor along each axis according to the tile map.
 * @param a The tensor to tile
 * @param tile_map The tile map
 * @return The tiled tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if any of the tile_map values are less than or equal to 0.
 */
Tensor pascal_tensor_tile(Tensor a, index_t tile_map[]);

/**
 * @brief Concatenate two tensors
 * \par
 * Concatenate two tensors along the specified axis.
 * @param a The first tensor
 * @param b The second tensor
 * @param axis The axis to concatenate along
 * @return The concatenated tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes are not compatible for concatenation.
 */
Tensor pascal_tensor_append(Tensor a, Tensor b, index_t axis);

/**
 * @brief Expand the dimensions of a tensor
 * \par
 * Expand the dimensions of a tensor by adding a new dimension at the specified index. The new dimension will have size 1.
 * @param a The tensor to expand
 * @param dim The index to add the new dimension
 * @return The expanded tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if dim is greater than the number of dimensions of the tensor.
 */
Tensor pascal_tensor_expand_dims(Tensor a, index_t dim);

/**
 * @brief Retrieve the diagonal of a tensor
 * \par
 * Retrieve the diagonal of a tensor. The tensor must be at least 2-dimensional, and the last two dimensions must be square.
 * @param a The tensor to retrieve the diagonal from
 * @return The diagonal tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the tensor is not at least 2-dimensional.
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the last two dimensions are not equal to each other.
 */
Tensor pascal_tensor_diag(Tensor a);

/**
 * @brief Clamp the elements of a tensor to a range
 * \par
 * Clamp the elements of a tensor to a range.
 * @param a The tensor to clamp
 * @param min The minimum value to clamp to
 * @param max The maximum value to clamp to
 * @return The clamped tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if max is less than min.
 */
Tensor pascal_tensor_clamp(Tensor a, double min, double max);

/**
 * @brief Flatten a tensor
 * \par
 * Flatten a tensor into a 1D tensor.
 * @param a The tensor to flatten
 * @return The flattened tensor
 */
Tensor pascal_tensor_flatten(Tensor a);

/**
 * @brief Add two tensors
 * \par
 * Add two tensors element-wise if the shapes are equal. If the shapes are not equal, the tensors must 	be broadcastable (in the NumPy sense).
 * @param a The first summand
 * @param b The second summan
 * @return The sum of the two tensors
 */
Tensor pascal_tensor_add(Tensor a, Tensor b);

/**
 * @brief Subtract two tensors
 * \par
 * Subtract two tensors element-wise if the shapes are equal. If the shapes are not equal, the tensors must be broadcastable (in the NumPy sense).
 * @param a The minuend
 * @param b The subtrahend
 * @return The difference of the two tensors
 */
Tensor pascal_tensor_subtract(Tensor a, Tensor b);

/**
 * @brief Multiply a tensor by a scalar
 * \par
 * Multiply a tensor by a scalar.
 * @param a The tensor to multiply
 * @param k The scalar to multiply by
 * @return The product of the tensor and the scalar
 */
Tensor pascal_tensor_scalar_multiply(Tensor a, double k);

/**
 * @brief Multiply two tensors
 * \par
 * Multiply two tensors element-wise if the shapes are equal. If the shapes are not equal, the tensors must be broadcastable (in the NumPy sense).
 * @param a The first factor
 * @param b The second factor
 * @return The product of the two tensors
 */
Tensor pascal_tensor_multiply(Tensor a, Tensor b);

/**
 * @brief Element-wise division of a tensor by another tensor
 * \par
 * Divide the first tensor by the second tensor element-wise if the shapes are equal. If the shapes are not equal, the tensors must be broadcastable (in the NumPy sense).
 * @param a The dividend
 * @param k The divisor
 * @return The quotient of the two tensors
 */
Tensor pascal_tensor_divide(Tensor a, Tensor b);

/**
 * @brief Element-wise reciprocal of a tensor
 * \par
 * Calculate the element-wise reciprocal of a tensor.
 * @param a The tensor to take the reciprocal of
 * @return The tensor with the reciprocal of each element
 */
Tensor pascal_tensor_reciprocal(Tensor a);

/**
 * @brief Map an arbitrary operation over a tensor element-wise
 * \par
 * Map an arbitrary operation over a tensor element-wise.
 * @param a The tensor to map the operation over
 * @param operation The operation to map over the tensor
 * @return The tensor with the operation mapped over it
 */
Tensor pascal_tensor_map(Tensor a, double (*operation)(double a));

// TODO: Tensor pascal_tensor_map_multiple(double (*operation)(double[]), index_t n, ...);

/**
 * @brief Element-wise square of a tensor
 * \par
 * Calculate the element-wise square of a tensor.
 * @param a The tensor to square
 * @return The tensor with the square of each element
 */
Tensor pascal_tensor_square(Tensor a);

/**
 * @brief Sum the elements of a tensor over the given axes
 * \par
 * Sum the elements of a tensor over the given axes.
 * @param a The tensor to sum
 * @param axes The axes to sum over
 * @param n_axes The number of axes to sum over
 * @return The tensor with the sum over the given axes
 */
Tensor pascal_tensor_sum(Tensor a, index_t axes[], index_t n_axes);

/**
 * @brief Sum the elements of a tensor over the axes specified by the mask
 * \par
 * Sum the elements of a tensor over the axes specified by the mask.
 * @param a The tensor to sum
 * @param axes_mask The mask of axes to sum over
 * @return The tensor with the sum over the axes specified by the mask
 */
Tensor pascal_tensor_sum_mask(Tensor a, bool axes_mask[]);

/**
 * @brief Sum all the elements of a tensor
 * \par
 * Sum all the elements of a tensor.
 * @param a The tensor to sum
 * @return The sum of all the elements of the tensor
 */
Tensor pascal_tensor_sum_all(Tensor a);

/**
 * @brief Product of all the elements of a tensor
 * \par
 * Calculate the product of all the elements of a tensor.
 * @param a The tensor to take the product of
 * @return The product of all the elements of the tensor
 */
Tensor pascal_tensor_prod_all(Tensor a);

/**
 * @brief Mean value of a tensor
 * \par
 * Calculate the mean value of all elements in a tensor.
 * @param a The tensor to find the mean value of
 * @return The mean value of the tensor
 */
Tensor pascal_tensor_mean_all(Tensor a);

/**
 * @brief Dot product of two tensors
 * \par
 * Calculate the dot product of two tensors. One of the last two dimensions of the tensors must be 1, and the other must be equal.
 * The dot product is calculated on the last two dimensions of the tensors. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The first tensor
 * @param b The second tensor
 * @return The dot product of the two tensors
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes are not compatible for taking the dot product.
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 */
Tensor pascal_tensor_dot(Tensor a, Tensor b);

/**
 * @brief Matrix multiplication of two tensors
 * \par
 * Calculate the matrix multiplication of two tensors. The last dimension of the first tensor must be equal to the second-to-last dimension of the second tensor.
 * The matrix multiplication is calculated on the last two dimensions of the tensors. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The first tensor
 * @param b The second tensor
 * @return The matrix multiplication of the two tensors
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes are not compatible for taking the matrix multiplication.
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 */
Tensor pascal_tensor_matmul(Tensor a, Tensor b);

/**
 * @brief Solve a linear system of equations
 * \par
 * Solve a linear system of equations Ax = y for x. The tensor a must be square.
 * The solver runs on the last two dimensions of the tensors. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The matrix A
 * @param y The vector y
 * @return The solution x
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the matrix a is not square (in the last two dimensions).
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the matrix is singular.
 */
Tensor pascal_tensor_linalg_solve(Tensor a, Tensor y);

/**
 * @brief Invert a matrix
 * \par
 * Invert a matrix. The tensor a must be square.
 * The inversion runs on the last two dimensions of the tensor. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The matrix to invert
 * @return The inverted matrix
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the matrix a is not square (in the last two dimensions).
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the matrix is singular.
 */
Tensor pascal_tensor_linalg_inv(Tensor a);

/**
 * @brief Calculate the cholesky decomposition of a matrix
 * \par
 * Calculate the cholesky decomposition of a matrix. The tensor a must be square and positive definite.
 * The cholesky decomposition runs on the last two dimensions of the tensor. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The matrix to decompose
 * @return The cholesky decomposition of the matrix
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the matrix a is not square (in the last two dimensions).
 */
Tensor pascal_tensor_linalg_cholesky(Tensor a);

/**
 * @brief Solve a linear system of equations with a triangular matrix
 * \par
 * Solve a linear system of equations Ax = y for x, where A is a triangular matrix.
 * The solver runs on the last two dimensions of the tensors. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The triangular matrix A
 * @param y The vector y
 * @param lower Whether the matrix is lower triangular (true) or upper triangular (false)
 * @return The solution x
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the matrix a is not square (in the last two dimensions).
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 * @note This function does not check if the matrix is triangular.
 */
Tensor pascal_tensor_linalg_triangular_solve(Tensor a, Tensor y, bool lower);

/**
 * @brief Convolve a tensor with a filter
 * \par
 * Convolve a tensor with a filter. The filter must be smaller than the tensor in all dimensions.
 * The convolution runs on the last dimensions (given the number of dimensions of the filter) of the tensors. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The tensor to convolve
 * @param filter The filter to convolve with
 * @param stride The stride of the convolution
 *
 */
Tensor pascal_tensor_convolution(Tensor a, Tensor filter, index_t stride[]);

/**
 * @brief Convolve a 4D tensor with a 4D filter
 * \par
 * Convolve a tensor with a filter. The filter must be smaller than the tensor in all dimensions.
 * The convolution runs on the last four dimensions of the tensors. The other dimensions are broadcasted (in the NumPy sense).
 * @param a The tensor to convolve
 * @param filter The filter to convolve with
 * @param stride The stride of the convolution
 * @return The convolved tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the filter tensor does not have 4 dimensions.
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the number of channels in the input tensor is not equal to the number of channels in the filter tensor.
 */
Tensor pascal_tensor_conv2d(Tensor a, Tensor filter, index_t stride[]);

/**
 * @brief Run a max pooling operation on a tensor
 * \par
 * Run a max pooling operation on a tensor. The filter must have less or equal dimensions than the input tensor.
 * @param a The tensor to max pool
 * @param filter_shape The shape of the filter
 * @param stride The stride of the max pooling
 * @param filter_ndim The number of dimensions of the filter
 * @return The max pooled tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the filter tensor has more dimensions than the input tensor.
 */
Tensor pascal_tensor_max_pool(Tensor a, index_t filter_shape[], index_t stride[], index_t filter_ndim);

/**
 * @brief Generate a random tensor with uniformly distributed values
 * \par
 * Generate a random tensor of the given shape with uniformly distributed values between min and max.
 * @param min The minimum value of the distribution
 * @param max The maximum value of the distribution
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The sampled tensor
 */
Tensor pascal_tensor_random_uniform(double min, double max, index_t shape[], index_t ndim);

/**
 * @brief Generate a random tensor with normally distributed values
 * \par
 * Generate a random tensor of the given shape with normally distributed values with the given mean and variance.
 * @param mean The mean of the distribution
 * @param variance The variance of the distribution
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The sampled tensor
 */
Tensor pascal_tensor_random_normal(double mean, double variance, index_t shape[], index_t ndim);

#if TENSOR_USE_UXHW || (!TENSOR_USE_UXHW && !(TENSOR_BACKEND == TENSOR_BACKEND_GSL))
/**
 * @brief Create a 'uncertain' tensor with normally distributed values.
 * \par
 * When using UxHw, the values of the tensor will represent Gaussian random variables with the given means and stddevs.
 * @param means The means of the distribution. The size of this array must match the shape.
 * @param stddevs The standard deviations of the distribution. The size of this array must match the shape.
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The uncertain tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the size of the means or stddevs arrays do not match the shape.
 */
Tensor pascal_tensor_uncertain_normal(double means[], double stddevs[], index_t shape[], index_t ndim);
#else
	#include <gsl/gsl_randist.h>
	#include <gsl/gsl_rng.h>
/**
 * @brief Create a 'uncertain' tensor with normally distributed values.
 * \par
 * When using GSL, the values of the tensor will be samples from a Gaussian random variables with the given means and stddevs. The returned tensor can therefore be used in a Monte Carlo simulation.
 * @param means The means of the distribution. The size of this array must match the shape.
 * @param stddevs The standard deviations of the distribution. The size of this array must match the shape.
 * @param shape The shape of the tensor
 * @param ndim The number of dimensions of the tensor
 * @param r The GSL random number generator
 * @return The uncertain tensor
 * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the size of the means or stddevs arrays do not match the shape.
 */
Tensor pascal_tensor_uncertain_normal(double means[], double stddevs[], index_t shape[], index_t ndim, gsl_rng* r);
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "pascal.h"

typedef struct broadcast_output_def {
	Tensor   tensor;
	index_t* a_stride;
	index_t* b_stride;
} BroadcastOutput_D, *BroadcastOutput;

BroadcastOutput pascal_tensor_broadcast_output_init();
void            pascal_tensor_broadcast_output_free(BroadcastOutput output);

/**
 * @brief Check if two tensors need to be broadcasted
 * \par
 * Check if two tensors need to be broadcasted.
 * @param a The first tensor
 * @param b The second tensor
 * @return True if the tensors need to be broadcasted, false otherwise
 */
bool pascal_tensor_broadcast_is_needed(Tensor a, Tensor b);

/**
 * @brief Check if two tensors need to be broadcasted for linear algebra operations
 * \par
 * Check if two tensors need to be broadcasted for linear algebra operations.
 * @param a The first tensor
 * @param b The second tensor
 * @return True if the tensors need to be broadcasted, false otherwise
 */
bool pascal_tensor_broadcast_is_needed_linalg(Tensor a, Tensor b);

/**
 * @brief Calculate the linear index of a tensor element from its index
 * \par
 * Calculate the linear index of a tensor element from its index and stride.
 * @param index The index of the element
 * @param stride The stride of the tensor
 * @param ndim The number of dimensions of the tensor
 * @return The linear index of the element
 */
index_t pascal_tensor_linear_index_from_index(index_t index[], index_t stride[], index_t ndim);

/**
 * @brief Calculate the BroadcastOutput of two tensors.
 * \par
 * Calculate the BroadcastOutput of two tensors. This generates the tensor with the broadcasted shape.
 * @param a The first tensor
 * @param b The second tensor
 * @return The BroadcastOutput of the two tensors
 * @throw  * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 */
BroadcastOutput pascal_tensor_broadcast(Tensor a, Tensor b);

/**
 * @brief Calculate the BroadcastOutput of two tensors for linear algebra operations.
 * \par
 * Calculate the BroadcastOutput of two tensors for linear algebra operations. This generates the tensor with the broadcasted shape.
 * @param a The first tensor
 * @param b The second tensor
 * @param out_shape The shape of the output tensor
 * @param out_ndim The number of dimensions of the output tensor
 * @return The BroadcastOutput of the two tensors
 * @throw  * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 */
BroadcastOutput pascal_tensor_broadcast_linalg(Tensor a, Tensor b, index_t out_shape[], index_t out_ndim);

/**
 * @brief Perform a broadcasted operation on two tensors.
 * \par
 * Broadcast two tensors and perform an operation on them.
 * @param a The first tensor
 * @param b The second tensor
 * @param operation The operation to perform on the tensors
 * @return The tensor with the result of the operation
 * @throw  * @throw If TENSOR_USE_ASSERT is set a fatal error is raised if the shapes aren't compatible for broadcasting.
 */
Tensor pascal_tensor_broadcast_and_operate(Tensor a, Tensor b, double (*operation)(double a, double b));

typedef struct pascal_tensor_iterator_def {
	index_t* indexes;
	index_t  offset;
	// TODO: Keep track of whether the end of the iterator had been reached.
} TensorIterator_D, *TensorIterator;

/**
 * @brief Create a new `TensorIterator` object.
 * \par
 * The `TensorIterator` object is used to iterate over a Tensor object. It can be used with other functions to obtain the current iterate or the next iterate.
 * This object must be freed using `pascal_tensor_iterator_free`.
 * @param a The Tensor object to iterate over.
 * @return TensorIterator The new `TensorIterator` object.
 * @see `pascal_tensor_iterate`
 * @see `pascal_tensor_iterate_next`
 * @see `pascal_tensor_iterate_current`
 */
TensorIterator pascal_tensor_iterator_new(Tensor a);

TensorIterator pascal_tensor_iterator_copy(TensorIterator iterator, index_t ndim);
void           pascal_tensor_iterator_free(TensorIterator iterator);

/**
 * @brief Iterate the `Tensor` object using the generated iterator.
 * \par
 * This function will iterate over the Tensor object using the generated iterator. The iterator will be updated to the next iterate.
 * If the iterator has reached the end, it will loop back to the beginning. Therefore, you should know how long the iteration should last.
 * Use `pascal_tensor_iterate_current` to get the current iterate and `pascal_tensor_iterate_next` to get the next iterate.
 * @param iterator The `TensorIterator` object.
 * @param a The Tensor object to iterate over.
 * @see `pascal_tensor_iterate_next`
 * @see `pascal_tensor_iterate_current`
 */
void pascal_tensor_iterate(TensorIterator iterator, Tensor a);

/**
 * @brief Iterate the tensor object and get the next value.
 * \par
 * This function will return the next iterate of the Tensor object. the `iterate` will be updated to the next iterate.
 * Use `pascal_tensor_iterate` to iterate over the `Tensor` object and `pascal_tensor_iterate_current` to get the current iterate.
 * @param iterator The `TensorIterator` object.
 * @param a The `Tensor` object to iterate over.
 * @return double The next iterate of the Tensor object.
 * @see `pascal_tensor_iterate`
 * @see `pascal_tensor_iterate_current`
 */
double pascal_tensor_iterate_next(TensorIterator iterator, Tensor a);

/**
 * @brief Get the current iterate of the Tensor object.
 * \par
 * This function will return the current iterate of the Tensor object. The iterate will not be updated.
 * Use `pascal_tensor_iterate` to iterate over the Tensor object and `pascal_tensor_iterate_next` to get the next iterate.
 * @param iterator The `TensorIterator` object.
 * @param a The Tensor object to iterate over.
 * @return double The current iterate of the Tensor object.
 * @see `pascal_tensor_iterate`
 * @see `pascal_tensor_iterate_next`
 */
double pascal_tensor_iterate_current(TensorIterator iterator, Tensor a);

/**
 * @brief Given a particular index, update the indexes with the next index in the usual order.
 * \par
 * This function will update the indexes with the next index in the usual order. The indexes will be updated in-place. For example, if you pass in [0, 1, 3] and the shape is [2, 3, 4], the indexes will be updated to [0, 2, 0]
 * @param indexes The indexes to update.
 * @param shape The shape of the tensor.
 * @param ndim The number of dimensions of the tensor.
 */
void     pascal_tensor_iterate_indexes_next(index_t* indexes, index_t* shape, index_t ndim);

double   pascal_tensor_random_sample_normal(double mean, double stddev);
double   pascal_tensor_random_sample_uniform(double min, double max);

void     pascal_tensor_print_values(double array[], index_t size);

bool     pascal_tensor_utils_shapes_equal(Tensor a, Tensor b);

index_t* pascal_tensor_utils_apply_transpose_map(index_t* indexes, index_t* transpose_map, index_t ndim);

index_t  pascal_tensor_utils_size_from_shape(index_t shape[], index_t ndim);

index_t* pascal_tensor_utils_default_stride(index_t shape[], index_t ndim);
index_t* pascal_tensor_utils_index_from_linear_index_transpose_safe(index_t linear_index, Tensor a);
void     pascal_tensor_utils_index_from_linear_index(index_t out[], index_t linear_index, index_t stride[], index_t ndim);
index_t* pascal_tensor_utils_get_masked_index(index_t indexes[], index_t shape[], index_t ndim, index_t broadcasted_ndim);

index_t  pascal_tensor_utils_get_masked_offset(index_t indexes[], Tensor a, index_t broadcasted_ndim);

double*  pascal_tensor_utils_linalg_get_array_col_maj(Tensor a, index_t indexes[]);

Tensor   pascal_tensor_utils_unravel(Tensor a);
void     pascal_tensor_utils_unravel_and_replace(Tensor a);
double*  pascal_tensor_utils_get_pointer(Tensor a, index_t indexes[]);

index_t* pascal_tensor_utils_get_convolution_start_index(index_t indexes[], index_t stride[], index_t indexes_ndim, index_t stride_ndim);
double*  pascal_tensor_utils_get_convolution_array(Tensor a, index_t filter_shape[], index_t filter_size, index_t filter_ndim, index_t start_index[]);

Tensor   pascal_tensor_binary_operate(Tensor a, Tensor b, double (*operation)(double, double));

Tensor   pascal_tensor_utils_unary_chain_rule(Tensor current_grad, Tensor a, double (*gradient_fn)(double));

int      load_data(char file_name[], double** x, double** y);
void     load_pascal_tensor_data(char file_name[], Tensor* x, Tensor* y, index_t num_data_points, index_t x_dim, index_t y_dim);

#endif
