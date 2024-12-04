N = 2000
PYTORCH_N_TRAINTING_ITER = 1000
CC=clang

OPTIMISATION_LEVEL=-O3

ENTRYPOINTS := \
	bnn-inference.c \

SOURCES := \

RESULTS_DIR=results
DATA_DIR=data
PLOTS_DIR=plots
BUILD_DIR=build
SRC_DIR=src
INCLUDE_DIR=include

INCLUDE_FLAGS=-I../../include -I$(INCLUDE_DIR)
CFLAGS+=$(OPTIMISATION_LEVEL)
LDFLAGS=-L../../libs/ -lpascal
FRAMEWORK_FLAGS=

OBJECTS := $(SOURCES:%=$(BUILD_DIR)/%.o)
LIB := $(OBJECTS)

ENTRYPOINTS_FLAT := $(ENTRYPOINTS:%.c=$(BUILD_DIR)/%)

.PHONY := \
	run \
	clean \

$(BUILD_DIR)/%.c.o: $(SRC_DIR)/%.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

.SECONDEXPANSION:
$(addsuffix .c.o,$(ENTRYPOINTS_FLAT)):: $$(patsubst $(BUILD_DIR)/%.c.o,$(SRC_DIR)/%.c,$$@)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $<  $(INCLUDE_FLAGS) -o $@

.SECONDEXPANSION:
$(ENTRYPOINTS_FLAT): $$@.c.o $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(FRAMEWORK_FLAGS)

build: $(ENTRYPOINTS_FLAT)


run: $(ENTRYPOINTS_FLAT)
	@echo ...............................................................................
	@echo Running bnn-inference
	python src/pytorch.py $(N) $(PYTORCH_N_TRAINTING_ITER)
	./$(BUILD_DIR)/bnn-inference $(N)
	python src/analyse.py
	@echo ...............................................................................

clean:
	rm -fv $(BUILD_DIR)/*
