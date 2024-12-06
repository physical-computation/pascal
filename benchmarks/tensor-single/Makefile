CC=clang
PYTHON=python

OPTIMISATION_LEVEL=-O3

SRC_DIR=src
PYTHON_SRC=python-src
BUILD_DIR=build
INCLUDE_DIR=include
RESULTS_DIR=results

INCLUDE_FLAGS=-I../../include -I$(INCLUDE_DIR)
CFLAGS+=$(OPTIMISATION_LEVEL)
LDFLAGS=-L../../libs/ -lpascal
FRAMEWORK_FLAGS=
DEBUG_FLAGS=-g

SOURCES := \

ENTRYPOINTS := \
	add \
	add_broadcast \
	multiply \
	multiply_broadcast \
	reciprocal \
	square \
	matmul \
	dot \
	linalg_inv \
	linalg_solve \
	sum \
	sum_all \


OBJECTS := $(SOURCES:%=$(BUILD_DIR)/%.o)
LIB := $(OBJECTS:%=%)
ENTRYPOINTS_FLAT := $(ENTRYPOINTS:%=$(BUILD_DIR)/%)

.PHONY := \
	run \
	clean \

$(BUILD_DIR)/%.c.o: $(SRC_DIR)/%.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

.SECONDEXPANSION:
$(addsuffix .c.o,$(ENTRYPOINTS_FLAT)): $$(patsubst $(BUILD_DIR)/%.c.o,$(SRC_DIR)/%.c,$$@)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $<  $(DEBUG_FLAGS) $(INCLUDE_FLAGS) -o $@

.SECONDEXPANSION:
$(ENTRYPOINTS_FLAT): $$@.c.o $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(DEBUG_FLAGS) $(FRAMEWORK_FLAGS)

.SECONDEXPANSION:
run: $(ENTRYPOINTS_FLAT)
	mkdir -p $(RESULTS_DIR)
	$(foreach ep, $(ENTRYPOINTS), $(BUILD_DIR)/$(ep); $(PYTHON) $(PYTHON_SRC)/$(ep).py; echo;)

clean:
	rm -fv $(BUILD_DIR)/*
