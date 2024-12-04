BUILD_SOURCES=build-sources.inc

# backend flags
TENSOR_BACKEND_GSL=0
TENSOR_BACKEND_BLAS=1
TENSOR_BACKEND_CLAPACK=2

# set backend
TENSOR_BACKEND=$(TENSOR_BACKEND_CLAPACK)

# general pascal flags
TENSOR_USE_UXHW=0
TENSOR_USE_SIMD=0
TENSOR_PRINT_VERBOSE=0
TENSOR_USE_ASSERT=0


# arbiter flags
ARBITER_VERBOSE=0
ARBITER_STDERR_LOG_DIR=

ENTRYPOINT=main

# source directories
SRC_DIR=src
PYTHON_SRC=python-src

# other directories
BUILD_DIR=build
INCLUDE_DIR=include
LIBS_DIR=libs
TESTS_DIR=tests
CONFIGS_DIR=configs
EXAMPLES_DIR=examples
BENCHMARKS_DIR=benchmarks
DOCS_DIR=docs

# get platform name. Should be darwin (macOs) or linx (Linux)
PLATFORM=$(shell uname -s | awk '{print tolower($$0)}')

LIB=$(LIBS_DIR)/libpascal.a

# default initial compile flags
INCLUDE_FLAGS= -I$(INCLUDE_DIR) -Iarbiter/include
LDFLAGS=-L$(LIBS_DIR) -lpascal

# default optimisation level and debug flags
OPTFLAGS=-O3
DEBUG_FLAGS=


CONFIGURATION=default
include $(CONFIGS_DIR)/$(CONFIGURATION).conf

CFLAGS+=-D'TENSOR_PRINT_VERBOSE=$(TENSOR_PRINT_VERBOSE)' -D'TESTS_VERBOSE=$(TESTS_VERBOSE)' -D'TESTS_STDERR_LOG_DIR="$(TESTS_STDERR_LOG_DIR)"' -D'TENSOR_USE_SIMD=$(TENSOR_USE_SIMD)' -D'TENSOR_USE_ASSERT=$(TENSOR_USE_ASSERT)' -D'TENSOR_USE_UXHW=$(TENSOR_USE_UXHW)' -D'TENSOR_BACKEND_GSL=$(TENSOR_BACKEND_GSL)' -D'TENSOR_BACKEND_BLAS=$(TENSOR_BACKEND_BLAS)' -D'TENSOR_BACKEND_CLAPACK=$(TENSOR_BACKEND_CLAPACK)'


include $(BUILD_SOURCES)

SOURCES_FLAT:= $(SOURCES:$(SRC_DIR)/%=%)
OBJECTS := $(SOURCES_FLAT:%=$(BUILD_DIR)/%.o)


TESTS := \
	$(TESTS_DIR)/load_data \
	$(TESTS_DIR)/tensor_integration \
	$(TESTS_DIR)/tensor \
	$(TESTS_DIR)/tensor_random_sample \
	$(TESTS_DIR)/tensor_utils \
	$(TESTS_DIR)/autodiff \
	$(TESTS_DIR)/autodiff_primitives \
	$(TESTS_DIR)/tensor_broadcast \
	$(TESTS_DIR)/tensor_iterate \

EXAMPLES := \
	bnn-simple \
	gp-simple \
	nn-simple \
	bnn-inference

EXPERIMENTS := \

BENCHMARKS := \
	tensor-single \

.PHONY := \
	default \
	lib \
	run \
	test \
	examples \
	benchmarks \
	clean \
	docs \

default: lib
lib: $(LIB)

$(LIB): Makefile $(OBJECTS)
	mkdir -p $(dir $@)
	ar r $@ $(filter-out $<,$^)

$(BUILD_DIR)/$(ENTRYPOINT).o: Makefile $(ENTRYPOINT).c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(DEBUG_FLAGS) $(INCLUDE_FLAGS) -c $(filter-out $<,$^) -o $@

$(BUILD_DIR)/%.o: Makefile $(SRC_DIR)/%.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(DEBUG_FLAGS) $(INCLUDE_FLAGS) -c $(filter-out $<,$^) -o $@


$(BUILD_DIR)/$(ENTRYPOINT): Makefile $(BUILD_DIR)/$(ENTRYPOINT).o $(LIB)
	$(LD) $(MAP) $(LD_SCRIPT_FLAGS) $(FRAMEWORK_FLAGS) $(filter-out $<,$^) -o $@ $(LDFLAGS)

run: $(BUILD_DIR)/$(ENTRYPOINT)
	./$(BUILD_DIR)/$(ENTRYPOINT)


$(BUILD_DIR)/arbiter/src/arbiter.o: Makefile arbiter/src/arbiter.c arbiter/include/arbiter.h
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(DEBUG_FLAGS) $(INCLUDE_FLAGS) -D'ARBITER_STDERR_LOG_DIR="$(ARBITER_STDERR_LOG_DIR)"' -D'ARBITER_VERBOSE=$(ARBITER_VERBOSE)' -c arbiter/src/arbiter.c -o $@

$(TESTS_DIR)/%/test: $(TESTS_DIR)/%/test.o $(BUILD_DIR)/arbiter/src/arbiter.o $(LIB)
	$(LD) $< $(BUILD_DIR)/arbiter/src/arbiter.o $(FRAMEWORK_FLAGS) -o $@ $(LDFLAGS)

$(TESTS_DIR)/%/test.o: $(TESTS_DIR)/%/test.c
	$(CC) $(OPTFLAGS) $(CFLAGS) -c $< $(INCLUDE_FLAGS) -o $@

test: $(TESTS:%=%/test)
	$(foreach executable,$^,$(executable);)


.SECONDEXPANSION:
$(addprefix eg-,$(EXAMPLES)): $(EXAMPLES_DIR)/$$(patsubst eg-%,%,$$@)/Makefile $(LIB)
	@cd $(EXAMPLES_DIR)/$(patsubst eg-%,%,$@); make run; echo

.SECONDEXPANSION:
$(addprefix bm-,$(BENCHMARKS)): $(BENCHMARKS_DIR)/$$(patsubst bm-%,%,$$@)/Makefile $(LIB)
	cd $(BENCHMARKS_DIR)/$(patsubst bm-%,%,$@); make run


examples: $(EXAMPLES:%=eg-%)
benchmarks: $(BENCHMARKS:%=bm-%)


clean:
	rm -rfv $(BUILD_DIR)/*
	rm -fv $(ENTRYPOINT) \
	 	$(LIB) \
		$(OBJECTS) \
		$(TESTS:%=%/test) \
		$(TESTS:%=%/test.o) \
		$(TESTS:%=$(TESTS_DIR)/%/test) \
		$(TESTS:%=$(TESTS_DIR)/%/test) \

	$(if $(EXAMPLES), cd $(EXAMPLES:%=$(EXAMPLES_DIR)/%); make clean, )
	$(if $(BENCHMARKS), cd $(BENCHMARKS:%=$(BENCHMARKS_DIR)/%); make clean, )

docs : $(DOCS_DIR)/config
	doxygen $(DOCS_DIR)/config
