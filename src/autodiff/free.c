#include <stdlib.h>

#include "pascal_autodiff.h"

static index_t append_to_nodes_list(AutodiffNode** nodes, AutodiffNode node, index_t n_nodes) {
	for (int i = 0; i < n_nodes; i++) {
		if ((*nodes)[i] == node) {
			return n_nodes;
		}
	}

	*nodes            = realloc((*nodes), sizeof(AutodiffNode) * (n_nodes + 1));
	(*nodes)[n_nodes] = node;

	return n_nodes + 1;
}

index_t pascal_autodiff_free_recurse(AutodiffNode** nodes, AutodiffNode node, index_t n_nodes) {
	n_nodes = append_to_nodes_list(nodes, node, n_nodes);

	if (node->forward != NULL) {
		pascal_tensor_free(node->forward);
		node->forward = NULL;
	}

	if (node->grad != NULL) {
		pascal_tensor_free(node->grad);
		node->grad = NULL;
	}

	if (node->_transform_info != NULL) {
		free(node->_transform_info);
		node->_transform_info = NULL;
	}

	if (node->next != NULL) {
		for (int i = 0; i < node->num_inputs; i++) {
			n_nodes = pascal_autodiff_free_recurse(nodes, node->next[i], n_nodes);
		}
		free(node->next);
		node->next = NULL;
	}

	return n_nodes;
}
void pascal_autodiff_free(AutodiffNode node) {
	index_t       n_nodes = 0;
	AutodiffNode* nodes   = malloc(sizeof(AutodiffNode) * 1);
	n_nodes               = pascal_autodiff_free_recurse(&nodes, node, n_nodes);

	for (int i = 0; i < n_nodes; i++) {
		free(nodes[i]);
	}
	free(nodes);
}
