#include "arbiter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pascal_autodiff.h"
#include "pascal_tensor_chain_rule.h"

#define NUM_TESTS 1

int main() {
	void (*tests[NUM_TESTS])() = {
			test_add_add_chain_rule,
	};

	arbiter_run_tests(NUM_TESTS, "pascal_tensor_chain_rule", tests);
}
