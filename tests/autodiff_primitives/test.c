#include "arbiter.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "_pascal_autodiff_primitives.h"

static void test_add() {
	double  values1[12]   = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	double  values2[12]   = {0.08251104, -0.6025303, 0.97252063, -0.54292459, 0.17257516, 0.79742578, -0.58548351, 0.14797305, -0.75000088, -0.63950701, 0.38753021, 0.50788599};
	index_t shape[3]      = {2, 2, 3};
	index_t ndim          = 3;

	Tensor a              = pascal_tensor_new(values1, shape, ndim);
	Tensor b              = pascal_tensor_new(values2, shape, ndim);

	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_add_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.84115815, 0.22260462, 1.08978058, -0.55310164, 0.09965402, -0.06391502, -1.24792457, 0.87534213, -0.30922915, -0.45503938, -0.33384304, 0.64823777};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12] = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad  = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient = malloc(sizeof(Tensor) * 2);

	out_gradient[0]      = _autodiff_primitive_add_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]      = _autodiff_primitive_add_gradient(inputs, out_forward, current_grad, 1);

	arbiter_assert(out_gradient[0]->ndim == expected_ndim);
	arbiter_assert(out_gradient[0]->size == expected_size);
	arbiter_assert(out_gradient[1]->ndim == expected_ndim);
	arbiter_assert(out_gradient[1]->size == expected_size);

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape[i]);
		arbiter_assert(out_gradient[0]->_stride[i] == expected_stride[i]);
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape[i]);
		arbiter_assert(out_gradient[1]->_stride[i] == expected_stride[i]);
	}

	double g1_expected_values[12] = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.29216220, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	double g2_expected_values[12] = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.29216220, 0.96526745, 0.02918029, 0.75881814, 0.69076706};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
		arbiter_assert(fabs(out_gradient[1]->values[i] - g2_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static void test_add_with_broadcast() {
	double  values1[18]   = {0.95856375, 0.00832376, 0.77430234, 0.8839322, 0.77050462, 0.39533107, -0.42940624, -0.754459, -0.11427941, 0.01886732, -0.59086918, -0.94458931, 0.04557025, -0.62257652, -0.60244149, 0.84349014, -0.72420908, -0.25368052};
	index_t shape1[4]     = {2, 3, 1, 3};
	index_t ndim1         = 4;

	double  values2[9]    = {-0.01175312, -0.41021609, 0.14492056, -0.80543126, -0.93982857, -0.06325477, 0.48700245, 0.44713226, 0.03564452};
	index_t shape2[3]     = {3, 3, 1};
	index_t ndim2         = 3;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);

	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_add_forward(inputs);

	index_t expected_ndim = 4;
	index_t expected_size = 54;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[4]  = {2, 3, 3, 3};
	index_t expected_stride[4] = {27, 9, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[54] = {0.94681063, -0.00342936, 0.76254922, 0.54834766, -0.40189233, 0.36408625, 1.10348431, 0.15324432, 0.91922290, 0.07850094, -0.03492664, -0.41010019, -0.05589637, -0.16932395, -0.54449750, 0.82067743, 0.70724985, 0.33207630, 0.05759621, -0.26745655, 0.37272304, 0.01772602, -0.30732674, 0.33285285, -0.39376172, -0.71881448, -0.07863489, 0.00711420, -0.60262230, -0.95634243, -0.39134877, -1.00108527, -1.35480540, 0.16378788, -0.44594862, -0.79966875, -0.75986101, -1.42800778, -1.40787275, -0.89425832, -1.56240509, -1.54227006, -0.01768452, -0.68583129, -0.66569626, 1.33049259, -0.23720663, 0.23332193, 1.29062240, -0.27707682, 0.19345174, 0.87913466, -0.68856456, -0.21803600};
	for (int i = 0; i < out_forward->size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[54]        = {0.91519959, 0.13800312, -0.09254871, -0.58235767, -0.5939935, -0.38685872, -0.48960586, 0.86925264, -0.83757308, 0.24233058, 0.16638243, -0.27179577, -0.92494878, 0.46406062, -0.69417506, 0.11408517, -0.97111489, -0.96515471, -0.18563716, -0.00632008, 0.12711179, 0.56649045, 0.60068747, 0.48857258, 0.58910655, 0.09023058, -0.36935144, -0.7527897, -0.09855495, 0.79261947, 0.35382456, -0.61033891, 0.44649717, -0.49084361, 0.60513559, 0.82155216, -0.50717096, -0.81094206, -0.55765998, -0.07710842, 0.69222272, -0.06457416, -0.30839276, 0.13838061, 0.18522591, 0.62411078, 0.08207344, -0.262256, -0.97001201, -0.10521724, 0.26772963, -0.7422678, -0.7468421, 0.88401914};
	Tensor current_grad         = pascal_tensor_new(values_cg, expected_shape, expected_ndim);

	Tensor* out_gradient        = malloc(sizeof(Tensor) * 2);

	out_gradient[0]             = _autodiff_primitive_add_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]             = _autodiff_primitive_add_gradient(inputs, out_forward, current_grad, 1);

	index_t expected_ndim_grad0 = 4;
	index_t expected_size_grad0 = 18;
	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad0);
	arbiter_assert(out_gradient[0]->size == expected_size_grad0);

	index_t expected_shape_grad0[4]  = {2, 3, 1, 3};
	index_t expected_stride_grad0[4] = {9, 3, 3, 1};
	for (int i = 0; i < out_gradient[0]->ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad0[i]);
		arbiter_assert(out_gradient[0]->_stride[i] == expected_stride_grad0[i]);
	}

	index_t expected_ndim_grad1 = 3;
	index_t expected_size_grad1 = 9;
	arbiter_assert(out_gradient[1]->ndim == expected_ndim_grad1);
	arbiter_assert(out_gradient[1]->size == expected_size_grad1);

	index_t expected_shape_grad1[3]  = {3, 3, 1};
	index_t expected_stride_grad1[3] = {3, 1, 1};
	for (int i = 0; i < out_gradient[1]->ndim; i++) {
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape_grad1[i]);
		arbiter_assert(out_gradient[1]->_stride[i] == expected_stride_grad1[i]);
	}

	double expected_values0[18] = {-0.15676394, 0.41326226, -1.31698051, -0.56853303, -0.34067184, -1.93112554, 0.96995984, 0.68459797, 0.24633293, -0.88980875, -0.10375827, 2.06066880, -0.89267214, 0.01966127, -0.43700823, -1.08816903, -0.76998590, 0.88949277};
	for (int i = 0; i < expected_size_grad0; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - expected_values0[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double expected_values1[9] = {0.90192882, -1.37322707, 0.47791784, -1.73885576, -0.60452308, -1.80697067, 0.37908277, 0.84825088, -0.29510507};
	for (int i = 0; i < expected_size_grad1; i++) {
		arbiter_assert(fabs(out_gradient[1]->values[i] - expected_values1[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static void test_subtract() {
	double  values1[12]   = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	double  values2[12]   = {0.08251104, -0.6025303, 0.97252063, -0.54292459, 0.17257516, 0.79742578, -0.58548351, 0.14797305, -0.75000088, -0.63950701, 0.38753021, 0.50788599};
	index_t shape[3]      = {2, 2, 3};
	index_t ndim          = 3;

	Tensor a              = pascal_tensor_new(values1, shape, ndim);
	Tensor b              = pascal_tensor_new(values2, shape, ndim);

	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_subtract_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.67613607, 1.42766522, -0.85526068, 0.53274754, -0.24549630, -1.65876658, -0.07695755, 0.57939603, 1.19077261, 0.82397464, -1.10890346, -0.36753421};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12] = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad  = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient = malloc(sizeof(Tensor) * 2);

	out_gradient[0]      = _autodiff_primitive_subtract_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]      = _autodiff_primitive_subtract_gradient(inputs, out_forward, current_grad, 1);

	arbiter_assert(out_gradient[0]->ndim == expected_ndim);
	arbiter_assert(out_gradient[0]->size == expected_size);
	arbiter_assert(out_gradient[1]->ndim == expected_ndim);
	arbiter_assert(out_gradient[1]->size == expected_size);

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape[i]);
		arbiter_assert(out_gradient[0]->_stride[i] == expected_stride[i]);
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape[i]);
		arbiter_assert(out_gradient[1]->_stride[i] == expected_stride[i]);
	}

	double g1_expected_values[12] = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.29216220, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	double g2_expected_values[12] = {0.59440882, -0.25243968, 0.31431052, -0.60317455, 0.80623455, -0.77578505, -0.27333454, 0.29216220, -0.96526745, -0.02918029, -0.75881814, -0.69076706};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
		arbiter_assert(fabs(out_gradient[1]->values[i] - g2_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static void test_subtract_with_broadcast() {
	double  values1[18]   = {0.95856375, 0.00832376, 0.77430234, 0.8839322, 0.77050462, 0.39533107, -0.42940624, -0.754459, -0.11427941, 0.01886732, -0.59086918, -0.94458931, 0.04557025, -0.62257652, -0.60244149, 0.84349014, -0.72420908, -0.25368052};
	index_t shape1[4]     = {2, 3, 1, 3};
	index_t ndim1         = 4;

	double  values2[9]    = {-0.01175312, -0.41021609, 0.14492056, -0.80543126, -0.93982857, -0.06325477, 0.48700245, 0.44713226, 0.03564452};
	index_t shape2[3]     = {3, 3, 1};
	index_t ndim2         = 3;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);

	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_subtract_forward(inputs);

	index_t expected_ndim = 4;
	index_t expected_size = 54;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[4]  = {2, 3, 3, 3};
	index_t expected_stride[4] = {27, 9, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[54] = {0.97031687, 0.02007688, 0.78605546, 1.36877984, 0.41853985, 1.18451843, 0.81364319, -0.13659680, 0.62938178, 1.68936346, 1.57593588, 1.20076233, 1.82376077, 1.71033319, 1.33515964, 0.94718697, 0.83375939, 0.45858584, -0.91640869, -1.24146145, -0.60128186, -0.87653850, -1.20159126, -0.56141167, -0.46505076, -0.79010352, -0.14992393, 0.03062044, -0.57911606, -0.93283619, 0.42908341, -0.18065309, -0.53437322, -0.12605324, -0.73578974, -1.08950987, 0.85100151, 0.18285474, 0.20298977, 0.98539882, 0.31725205, 0.33738708, 0.10882502, -0.55932175, -0.53918672, 0.35648769, -1.21121153, -0.74068297, 0.39635788, -1.17134134, -0.70081278, 0.80784562, -0.75985360, -0.28932504};
	for (int i = 0; i < out_forward->size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[54]        = {0.91519959, 0.13800312, -0.09254871, -0.58235767, -0.5939935, -0.38685872, -0.48960586, 0.86925264, -0.83757308, 0.24233058, 0.16638243, -0.27179577, -0.92494878, 0.46406062, -0.69417506, 0.11408517, -0.97111489, -0.96515471, -0.18563716, -0.00632008, 0.12711179, 0.56649045, 0.60068747, 0.48857258, 0.58910655, 0.09023058, -0.36935144, -0.7527897, -0.09855495, 0.79261947, 0.35382456, -0.61033891, 0.44649717, -0.49084361, 0.60513559, 0.82155216, -0.50717096, -0.81094206, -0.55765998, -0.07710842, 0.69222272, -0.06457416, -0.30839276, 0.13838061, 0.18522591, 0.62411078, 0.08207344, -0.262256, -0.97001201, -0.10521724, 0.26772963, -0.7422678, -0.7468421, 0.88401914};
	Tensor current_grad         = pascal_tensor_new(values_cg, expected_shape, expected_ndim);

	Tensor* out_gradient        = malloc(sizeof(Tensor) * 2);

	out_gradient[0]             = _autodiff_primitive_subtract_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]             = _autodiff_primitive_subtract_gradient(inputs, out_forward, current_grad, 1);

	index_t expected_ndim_grad0 = 4;
	index_t expected_size_grad0 = 18;
	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad0);
	arbiter_assert(out_gradient[0]->size == expected_size_grad0);

	index_t expected_shape_grad0[4]  = {2, 3, 1, 3};
	index_t expected_stride_grad0[4] = {9, 3, 3, 1};
	for (int i = 0; i < out_gradient[0]->ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad0[i]);
		arbiter_assert(out_gradient[0]->_stride[i] == expected_stride_grad0[i]);
	}

	index_t expected_ndim_grad1 = 3;
	index_t expected_size_grad1 = 9;
	arbiter_assert(out_gradient[1]->ndim == expected_ndim_grad1);
	arbiter_assert(out_gradient[1]->size == expected_size_grad1);

	index_t expected_shape_grad1[3]  = {3, 3, 1};
	index_t expected_stride_grad1[3] = {3, 1, 1};
	for (int i = 0; i < out_gradient[1]->ndim; i++) {
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape_grad1[i]);
		arbiter_assert(out_gradient[1]->_stride[i] == expected_stride_grad1[i]);
	}

	double expected_values0[18] = {-0.15676394, 0.41326226, -1.31698051, -0.56853303, -0.34067184, -1.93112554, 0.96995984, 0.68459797, 0.24633293, -0.88980875, -0.10375827, 2.06066880, -0.89267214, 0.01966127, -0.43700823, -1.08816903, -0.76998590, 0.88949277};
	for (int i = 0; i < expected_size_grad0; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - expected_values0[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double expected_values1[9] = {-0.90192882, 1.37322707, -0.47791784, 1.73885576, 0.60452308, 1.80697067, -0.37908277, -0.84825088, 0.29510507};
	for (int i = 0; i < expected_size_grad1; i++) {
		arbiter_assert(fabs(out_gradient[1]->values[i] - expected_values1[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static void test_multiply() {
	double  values1[12]   = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	double  values2[12]   = {0.08251104, -0.6025303, 0.97252063, -0.54292459, 0.17257516, 0.79742578, -0.58548351, 0.14797305, -0.75000088, -0.63950701, 0.38753021, 0.50788599};
	index_t shape[3]      = {2, 2, 3};
	index_t ndim          = 3;

	Tensor a              = pascal_tensor_new(values1, shape, ndim);
	Tensor b              = pascal_tensor_new(values2, shape, ndim);

	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_multiply_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.06259676, -0.49716879, 0.11403772, 0.00552537, -0.01258438, -0.68685536, 0.38784832, 0.10763102, -0.33057919, -0.11796834, -0.27955393, 0.07128270};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12] = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad  = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient = malloc(sizeof(Tensor) * 2);

	out_gradient[0]      = _autodiff_primitive_multiply_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]      = _autodiff_primitive_multiply_gradient(inputs, out_forward, current_grad, 1);

	arbiter_assert(out_gradient[0]->ndim == expected_ndim);
	arbiter_assert(out_gradient[0]->size == expected_size);
	arbiter_assert(out_gradient[1]->ndim == expected_ndim);
	arbiter_assert(out_gradient[1]->size == expected_size);

	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape[i]);
		arbiter_assert(out_gradient[0]->_stride[i] == expected_stride[i]);
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape[i]);
		arbiter_assert(out_gradient[1]->_stride[i] == expected_stride[i]);
	}

	double g1_expected_values[12] = {-0.04904529, -0.15210256, -0.30567346, -0.32747830, -0.13913606, 0.61863100, -0.16003287, -0.04323213, -0.72395144, -0.01866100, 0.29406495, 0.35083091};
	double g2_expected_values[12] = {-0.45094653, 0.20829680, -0.03685604, -0.00613854, 0.05879154, -0.66821532, -0.18106802, -0.21250975, 0.42546260, 0.00538282, -0.54739111, 0.09695039};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
		arbiter_assert(fabs(out_gradient[1]->values[i] - g2_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static void test_multiply_with_broadcast() {
	double  values1[18]   = {0.95856375, 0.00832376, 0.77430234, 0.8839322, 0.77050462, 0.39533107, -0.42940624, -0.754459, -0.11427941, 0.01886732, -0.59086918, -0.94458931, 0.04557025, -0.62257652, -0.60244149, 0.84349014, -0.72420908, -0.25368052};
	index_t shape1[4]     = {2, 3, 1, 3};
	index_t ndim1         = 4;

	double  values2[9]    = {-0.01175312, -0.41021609, 0.14492056, -0.80543126, -0.93982857, -0.06325477, 0.48700245, 0.44713226, 0.03564452};
	index_t shape2[3]     = {3, 3, 1};
	index_t ndim2         = 3;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);

	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_multiply_forward(inputs);

	index_t expected_ndim = 4;
	index_t expected_size = 54;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[4]  = {2, 3, 3, 3};
	index_t expected_stride[4] = {27, 9, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[54] = {-0.01126611, -0.00009783, -0.00910047, -0.39321827, -0.00341454, -0.31763128, 0.13891560, 0.00120628, 0.11221233, -0.71194663, -0.62058851, -0.31841200, -0.83074474, -0.72414226, -0.37154343, -0.05591293, -0.04873809, -0.02500658, -0.20912189, -0.36742338, -0.05565435, -0.19200138, -0.33734296, -0.05109801, -0.01530598, -0.02689233, -0.00407343, -0.00022175, 0.00694456, 0.01110187, -0.00773968, 0.24238404, 0.38748573, 0.00273426, -0.08562909, -0.13689041, -0.03670370, 0.50144259, 0.48522521, -0.04282822, 0.58511520, 0.56619172, -0.00288254, 0.03938093, 0.03810730, 0.41078176, -0.35269160, -0.12354303, 0.37715165, -0.32381724, -0.11342874, 0.03006580, -0.02581409, -0.00904232};
	for (int i = 0; i < out_forward->size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[54]        = {0.91519959, 0.13800312, -0.09254871, -0.58235767, -0.5939935, -0.38685872, -0.48960586, 0.86925264, -0.83757308, 0.24233058, 0.16638243, -0.27179577, -0.92494878, 0.46406062, -0.69417506, 0.11408517, -0.97111489, -0.96515471, -0.18563716, -0.00632008, 0.12711179, 0.56649045, 0.60068747, 0.48857258, 0.58910655, 0.09023058, -0.36935144, -0.7527897, -0.09855495, 0.79261947, 0.35382456, -0.61033891, 0.44649717, -0.49084361, 0.60513559, 0.82155216, -0.50717096, -0.81094206, -0.55765998, -0.07710842, 0.69222272, -0.06457416, -0.30839276, 0.13838061, 0.18522591, 0.62411078, 0.08207344, -0.262256, -0.97001201, -0.10521724, 0.26772963, -0.7422678, -0.7468421, 0.88401914};
	Tensor current_grad         = pascal_tensor_new(values_cg, expected_shape, expected_ndim);

	Tensor* out_gradient        = malloc(sizeof(Tensor) * 2);

	out_gradient[0]             = _autodiff_primitive_multiply_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]             = _autodiff_primitive_multiply_gradient(inputs, out_forward, current_grad, 1);

	index_t expected_ndim_grad0 = 4;
	index_t expected_size_grad0 = 18;
	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad0);
	arbiter_assert(out_gradient[0]->size == expected_size_grad0);

	index_t expected_shape_grad0[4]  = {2, 3, 1, 3};
	index_t expected_stride_grad0[4] = {9, 3, 3, 1};
	for (int i = 0; i < out_gradient[0]->ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad0[i]);
		arbiter_assert(out_gradient[0]->_stride[i] == expected_stride_grad0[i]);
	}

	index_t expected_ndim_grad1 = 3;
	index_t expected_size_grad1 = 9;
	arbiter_assert(out_gradient[1]->ndim == expected_ndim_grad1);
	arbiter_assert(out_gradient[1]->size == expected_size_grad1);

	index_t expected_shape_grad1[3]  = {3, 3, 1};
	index_t expected_stride_grad1[3] = {3, 1, 1};
	for (int i = 0; i < out_gradient[1]->ndim; i++) {
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape_grad1[i]);
		arbiter_assert(out_gradient[1]->_stride[i] == expected_stride_grad1[i]);
	}

	double expected_values0[18] = {0.15718208, 0.36801630, 0.03840185, 0.66689623, -0.50871939, 0.93236900, 0.18388882, 0.26872508, 0.26719496, -0.20743023, 0.33922576, -0.07341628, 0.50046735, -0.00616584, 0.49812900, -0.15623796, -0.03369688, 0.02350168};
	for (int i = 0; i < expected_size_grad0; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - expected_values0[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double expected_values1[9] = {0.10209525, -0.91716710, -2.25346404, 1.05267201, -1.13003344, -1.24075443, 0.60347793, -1.56219647, -0.58831621};
	for (int i = 0; i < expected_size_grad1; i++) {
		arbiter_assert(fabs(out_gradient[1]->values[i] - expected_values1[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static void test_matmul_simple() {
	double  values1[10]   = {0.55553049, -0.44091556, 0.00639342, 0.96838783, -0.6447367, -0.00753054, -0.48480336, -0.04607312, 0.08174824, 0.05052};
	index_t shape1[2]     = {2, 5};
	index_t ndim1         = 2;

	double  values2[10]   = {0.62050035, -0.99299838, -0.69053155, 0.28408452, -0.80742648, 0.90706307, 0.84921502, 0.26675885, -0.98595933, 0.12530195};
	index_t shape2[2]     = {5, 2};
	index_t ndim2         = 2;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);
	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_matmul_forward(inputs);

	index_t expected_ndim = 2;
	index_t expected_size = 4;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[2]  = {2, 2};
	index_t expected_stride[2] = {2, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[4] = {2.10206441, -0.49355967, 0.38691114, -0.14390122};
	for (int i = 0; i < out_forward->size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]        = {-0.59440882, 0.25243968, -0.31431052, 0.60317455};
	Tensor current_grad         = pascal_tensor_new(values_cg, expected_shape, expected_ndim);

	Tensor* out_gradient        = malloc(sizeof(Tensor) * 2);

	out_gradient[0]             = _autodiff_primitive_matmul_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]             = _autodiff_primitive_matmul_gradient(inputs, out_forward, current_grad, 1);

	index_t expected_ndim_grad0 = 2;
	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad0);

	index_t expected_shape_grad0[2] = {2, 5};
	for (int i = 0; i < out_gradient[0]->ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad0[i]);
	}

	index_t expected_ndim_grad1 = 2;
	arbiter_assert(out_gradient[1]->ndim == expected_ndim_grad1);

	index_t expected_shape_grad1[4] = {5, 2};
	for (int i = 0; i < out_gradient[1]->ndim; i++) {
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape_grad1[i]);
	}

	double expected_values0[10] = {-0.61950307, 0.48217225, 0.70892013, -0.43744038, 0.61769411, -0.79398114, 0.38839388, 0.80090000, -0.10601507, 0.38547634};
	for (int i = 0; i < 10; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - expected_values0[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double expected_values1[10] = {-0.32784530, 0.13569571, 0.41446289, -0.40372563, 0.01068096, -0.02617618, -0.60131260, 0.29376797, 0.36735821, -0.13228475};
	for (int i = 0; i < 10; i++) {
		arbiter_assert(fabs(out_gradient[1]->values[i] - expected_values1[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static void test_matmul_with_broadcast() {
	double  values1[24]   = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape1[4]     = {2, 3, 2, 2};
	index_t ndim1         = 4;

	double  values2[12]   = {50.0, 60.0, 50.0, 60.0, 50.0, 60.0, 50.0, 60.0, 50.0, 60.0, 50.0, 60.0};
	index_t shape2[3]     = {3, 2, 2};
	index_t ndim2         = 3;

	Tensor a              = pascal_tensor_new(values1, shape1, ndim1);
	Tensor b              = pascal_tensor_new(values2, shape2, ndim2);
	Tensor inputs[2]      = {a, b};

	Tensor out_forward    = _autodiff_primitive_matmul_forward(inputs);

	index_t expected_ndim = 4;
	arbiter_assert(out_forward->ndim == expected_ndim);

	index_t expected_shape[4] = {2, 3, 2, 2};
	for (int i = 0; i < out_forward->ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
	}
	double expected_values[24] = {150.0, 180.0, 350.0, 420.0, 550.0, 660.0, 150.0, 180.0, 350.0, 420.0, 550.0, 660.0, 150.0, 180.0, 350.0, 420.0, 550.0, 660.0, 150.0, 180.0, 350.0, 420.0, 550.0, 660.0};
	for (int i = 0; i < out_forward->size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	Tensor* out_gradient        = malloc(sizeof(Tensor) * 2);
	Tensor  current_grad        = pascal_tensor_ones(out_forward->shape, out_forward->ndim);

	out_gradient[0]             = _autodiff_primitive_matmul_gradient(inputs, out_forward, current_grad, 0);
	out_gradient[1]             = _autodiff_primitive_matmul_gradient(inputs, out_forward, current_grad, 1);

	index_t expected_ndim_grad0 = 4;
	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad0);

	index_t expected_shape_grad0[4] = {2, 3, 2, 2};
	for (int i = 0; i < out_gradient[0]->ndim; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad0[i]);
	}

	index_t expected_ndim_grad1 = 3;
	arbiter_assert(out_gradient[1]->ndim == expected_ndim_grad1);

	index_t expected_shape_grad1[3] = {3, 2, 2};
	for (int i = 0; i < out_gradient[1]->ndim; i++) {
		arbiter_assert(out_gradient[1]->shape[i] == expected_shape_grad1[i]);
	}

	double expected_gradient_0[24] = {110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110.};
	for (int i = 0; i < out_gradient[0]->size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - expected_gradient_0[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double expected_gradient_1[12] = {8., 8., 12., 12., 12., 12., 16., 16., 16., 16., 20., 20.};
	for (int i = 0; i < out_gradient[1]->size; i++) {
		arbiter_assert(fabs(out_gradient[1]->values[i] - expected_gradient_1[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(b);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	pascal_tensor_free(out_gradient[1]);
	free(out_gradient);
}

static double reciprocal_grad(double x) {
	return -1.0 * pow(x * x, -1);
}

static void test_reciprocal() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_reciprocal_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {1.31813591, 1.21192301, 8.52806094, -98.26030136, -13.71344441, -1.16098065, -1.50956826, 1.37481786, 2.26874804, 5.42100530, -1.38624492, 7.12495417};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_reciprocal_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {1.03277479, -0.37077265, 22.85912002, -5823.70265040, 151.61930664, -1.04566210, -0.62287375, 0.55222283, -4.96844169, -0.85752989, -1.45820182, -35.06677039};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static void test_exp() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_exp_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {2.13538532, 2.28218866, 1.12441168, 0.98987456, 0.92967414, 0.42259509, 0.51559121, 2.06962841, 1.55390595, 1.20257805, 0.48608428, 1.15067851};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_exp_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {-1.26929187, 0.57611497, -0.35341442, 0.59706714, -0.74953541, 0.32784295, 0.14092889, -0.60466719, 1.49993484, 0.03509158, 0.36884957, 0.79485081};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static double log_grad(double x) {
	return 1 / x;
}

static void test_log() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, 0.01017705, 0.07292114, 0.8613408, 0.66244106, 0.72736908, 0.44077173, 0.18446763, 0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_log_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {-0.27621855, -0.19220837, -2.14336201, -4.58762009, -2.61837670, -0.14926503, -0.41182369, -0.31832125, -0.81922816, -1.69028128, -0.32659859, -1.96360329};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_log_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {-0.78351161, 0.30593746, -2.68045927, 59.26811306, -11.05625269, 0.90067143, 0.41261715, -0.40166981, 2.18994864, 0.15818651, 1.05190779, 4.92168364};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static double square(double x) {
	return x * x;
}

static double square_grad(double x) {
	return 2 * x;
}

static void test_square() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_square_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.57554544, 0.68084764, 0.01374990, 0.00010357, 0.00531749, 0.74190797, 0.43882816, 0.52906578, 0.19427972, 0.03402831, 0.52037937, 0.01969862};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_square_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {-0.90189307, 0.41659359, -0.07371207, -0.01227708, 0.11758308, -1.33643063, -0.36213604, -0.42501950, 0.85092521, 0.01076564, -1.09478222, 0.19390077};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static void test_sum_all() {
	double  values[12]        = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape[3]          = {3, 2, 2};
	index_t ndim              = 3;

	Tensor a                  = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]          = {a};

	Tensor out_forward        = _autodiff_primitive_sum_all_forward(inputs);

	index_t expected_shape[1] = {1};
	for (int i = 0; i < out_forward->ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
	}

	for (int i = 0; i < expected_shape[0]; i++) {
		arbiter_assert(fabs(pascal_tensor_get(out_forward, (index_t[]){i}) - 42.0) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	Tensor* out_gradient           = malloc(sizeof(Tensor));
	Tensor  current_grad           = pascal_tensor_ones(shape, ndim);

	out_gradient[0]                = _autodiff_primitive_sum_all_gradient(inputs, out_forward, current_grad, 0);

	index_t expected_ndim_grad     = ndim;
	index_t expected_shape_grad[3] = {3, 2, 2};

	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad);
	for (int i = 0; i < expected_ndim_grad; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad[i]);
	}

	for (int i = 0; i < expected_shape_grad[0]; i++) {
		for (int j = 0; j < expected_shape_grad[1]; j++) {
			for (int k = 0; k < expected_shape_grad[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(out_gradient[0], (index_t[]){i, j, k}) - 1.0) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static void test_prod_all() {
	double  values[12]        = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	index_t shape[3]          = {3, 2, 2};
	index_t ndim              = 3;

	Tensor a                  = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]          = {a};

	Tensor out_forward        = _autodiff_primitive_prod_all_forward(inputs);

	index_t expected_shape[1] = {1};
	for (int i = 0; i < out_forward->ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
	}

	for (int i = 0; i < expected_shape[0]; i++) {
		arbiter_assert(fabs(pascal_tensor_get(out_forward, (index_t[]){i}) - 518400.0) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	Tensor* out_gradient           = malloc(sizeof(Tensor));
	Tensor  current_grad           = pascal_tensor_ones(shape, ndim);

	out_gradient[0]                = _autodiff_primitive_prod_all_gradient(inputs, out_forward, current_grad, 0);

	index_t expected_ndim_grad     = ndim;
	index_t expected_shape_grad[3] = {3, 2, 2};

	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad);
	for (int i = 0; i < expected_ndim_grad; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad[i]);
	}

	double expected_grad_values[12] = {518400.0, 259200.0, 172800.0, 129600.0, 103680.0, 86400.0, 518400.0, 259200.0, 172800.0, 129600.0, 103680.0, 86400.0};
	for (int i = 0; i < expected_shape_grad[0]; i++) {
		for (int j = 0; j < expected_shape_grad[1]; j++) {
			for (int k = 0; k < expected_shape_grad[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(out_gradient[0], (index_t[]){i, j, k}) - expected_grad_values[i * 4 + j * 2 + k]) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static void test_mean_all() {
	double  values[12]        = {0.03992382, -4.45962422, -0.42867344, 1.76058905, -1.31076798, -1.29633849, -1.8428066, 2.91669712, 4.10416207, -4.38678569, 0.76771607, -2.7582438};
	index_t shape[3]          = {3, 2, 2};
	index_t ndim              = 3;

	Tensor a                  = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]          = {a};

	Tensor out_forward        = _autodiff_primitive_mean_all_forward(inputs);

	index_t expected_shape[1] = {1};
	for (int i = 0; i < out_forward->ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
	}

	for (int i = 0; i < expected_shape[0]; i++) {
		arbiter_assert(fabs(pascal_tensor_get(out_forward, (index_t[]){i}) + 0.5745126742) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	Tensor* out_gradient           = malloc(sizeof(Tensor));
	Tensor  current_grad           = pascal_tensor_ones(shape, ndim);

	out_gradient[0]                = _autodiff_primitive_mean_all_gradient(inputs, out_forward, current_grad, 0);

	index_t expected_ndim_grad     = ndim;
	index_t expected_shape_grad[3] = {3, 2, 2};

	arbiter_assert(out_gradient[0]->ndim == expected_ndim_grad);
	for (int i = 0; i < expected_ndim_grad; i++) {
		arbiter_assert(out_gradient[0]->shape[i] == expected_shape_grad[i]);
	}

	for (int i = 0; i < expected_shape_grad[0]; i++) {
		for (int j = 0; j < expected_shape_grad[1]; j++) {
			for (int k = 0; k < expected_shape_grad[2]; k++) {
				arbiter_assert(fabs(pascal_tensor_get(out_gradient[0], (index_t[]){i, j, k}) - 0.08333333) < ARBITER_FLOATINGPOINT_ACCURACY);
			}
		}
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

static double sigmoid_grad(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

static void test_sigmoid() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_sigmoid_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.68105993, 0.69532525, 0.52928144, 0.49745576, 0.48177779, 0.29705929, 0.34019147, 0.67422767, 0.60844290, 0.54598658, 0.32709066, 0.53503046};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_sigmoid_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {-0.12911588, 0.05347885, -0.07830814, 0.15078973, -0.20129093, 0.16199561, 0.06135301, -0.06417188, 0.22996545, 0.00723336, 0.16701766, 0.17184410};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static double tanh_forward(double x) {
	return tanhl((long double)x);
}
static double tanh_grad(double x) {
	return 1 - tanh_forward(x) * tanh_forward(x);
}

static void test_tanh() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_tanh_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.64027939, 0.67785503, 0.11672545, -0.01017670, -0.07279216, -0.69694784, -0.57998563, 0.62145314, 0.41428392, 0.18240335, -0.61775920, 0.13943741};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_tanh_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {-0.35072635, 0.13644682, -0.31002809, 0.60311208, -0.80196256, 0.39895809, 0.18138936, -0.17932799, 0.79959747, 0.02820943, 0.46923308, 0.67733662};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static double relu(double x) {
	return x > 0 ? x : 0;
}

static double relu_grad(double x) {
	return x > 0 ? 1 : 0;
}

static void test_relu() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_relu_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.75864711, 0.82513492, 0.11725995, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.72736908, 0.44077173, 0.18446763, 0.00000000, 0.14035178};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_relu_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {-0.59440882, 0.25243968, -0.31431052, 0.00000000, 0.00000000, 0.00000000, 0.00000000, -0.29216220, 0.96526745, 0.02918029, 0.00000000, 0.69076706};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static void test_sin() {
	double  values[12]    = {0.75864711, 0.82513492, 0.11725995, -0.01017705, -0.07292114, -0.8613408, -0.66244106, 0.72736908, 0.44077173, 0.18446763, -0.72137325, 0.14035178};
	index_t ndim          = 3;
	index_t shape[3]      = {2, 2, 3};

	Tensor a              = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]      = {a};

	Tensor out_forward    = _autodiff_primitive_sin_forward(inputs);

	index_t expected_ndim = 3;
	index_t expected_size = 12;
	arbiter_assert(out_forward->ndim == expected_ndim);
	arbiter_assert(out_forward->size == expected_size);

	index_t expected_shape[3]  = {2, 2, 3};
	index_t expected_stride[3] = {6, 3, 1};
	for (int i = 0; i < expected_ndim; i++) {
		arbiter_assert(out_forward->shape[i] == expected_shape[i]);
		arbiter_assert(out_forward->_stride[i] == expected_stride[i]);
	}

	double expected_values[12] = {0.68794019, 0.73463933, 0.11699142, -0.01017687, -0.07285653, -0.75871667, -0.61504344, 0.66490684, 0.42663756, 0.18342322, -0.66041647, 0.13989144};
	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[12]          = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745, 0.02918029, 0.75881814, 0.69076706};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);

	Tensor* out_gradient          = malloc(sizeof(Tensor) * 2);

	out_gradient[0]               = _autodiff_primitive_sin_gradient(inputs, out_forward, current_grad, 0);
	double g1_expected_values[12] = {-0.43140253, 0.17126968, -0.31215213, 0.60314331, -0.80409193, 0.50536249, 0.21552243, -0.21822363, 0.87300978, 0.02868522, 0.56979618, 0.68397464};

	for (int i = 0; i < expected_size; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

static void test_linalg_inv() {
	index_t ndim       = 2;
	index_t shape[2]   = {3, 3};
	double  values[9]  = {-0.73755023, -0.26606191, -0.52607579, 0.00322696, -0.39497577, -0.69726951, 0.08398574, 0.38625244, 0.97182299};

	Tensor a           = pascal_tensor_new(values, shape, ndim);
	Tensor inputs[1]   = {a};

	Tensor out_forward = _autodiff_primitive_linalg_inv_forward(inputs);

	arbiter_assert(out_forward->ndim == ndim);
	for (int i = 0; i < ndim; i++) {
		arbiter_assert(out_forward->shape[i] == a->shape[i]);
		arbiter_assert(out_forward->_stride[i] == a->_stride[i]);
	}

	double expected_values[9] = {-1.38354971, 0.66887906, -0.26904388, -0.7453471, -8.12538337, -6.23332768, 0.41580677, 3.17164019, 3.529689992};
	for (int i = 0; i < a->size; i++) {
		arbiter_assert(fabs(out_forward->values[i] - expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	double values_cg[9]           = {-0.59440882, 0.25243968, -0.31431052, 0.60317455, -0.80623455, 0.77578505, 0.27333454, -0.2921622, 0.96526745};
	Tensor current_grad           = pascal_tensor_new(values_cg, shape, ndim);
	// Tensor current_grad = pascal_tensor_scalar_multiply(pascal_tensor_ones(shape, ndim), 2);

	Tensor* out_gradient          = malloc(sizeof(Tensor));

	out_gradient[0]               = _autodiff_primitive_linalg_inv_gradient(inputs, out_forward, current_grad, 0);

	double g1_expected_values[12] = {0.655400557324, 3.028541552287, -1.525818708102, -10.935228432457, 22.249331205961, -4.345595468784, -6.633634589262, 21.561179778852, -6.613203641776};

	for (int i = 0; i < 9; i++) {
		arbiter_assert(fabs(out_gradient[0]->values[i] - g1_expected_values[i]) < ARBITER_FLOATINGPOINT_ACCURACY);
	}

	pascal_tensor_free(a);
	pascal_tensor_free(out_forward);
	pascal_tensor_free(current_grad);
	pascal_tensor_free(out_gradient[0]);
	free(out_gradient);
}

#define NUM_TESTS 19

int main() {
	void (*tests[NUM_TESTS])() = {
			test_add,
			test_add_with_broadcast,
			test_subtract,
			test_subtract_with_broadcast,
			test_multiply,
			test_multiply_with_broadcast,
			test_matmul_simple,
			test_matmul_with_broadcast,
			test_reciprocal,
			test_exp,
			test_log,
			test_square,
			test_sum_all,
			test_prod_all,
			test_mean_all,
			test_sigmoid,
			test_tanh,
			test_relu,
			test_sin,
			// test_linalg_inv,
	};

	arbiter_run_tests(NUM_TESTS, "autodiff_primitives", tests);
}
