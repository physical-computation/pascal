LEARNING_RATE = 0.001
N = 2000
N_TEST_DATA_POINTS = 50
N_REPETITIONS = 200
N_DATA_PONTS = 50
N_NODES = 5
N_LAYERS = 2
PRINT_FREQUENCY = 200

CC=clang
PYTHON=python

OPTIMISATION_LEVEL=-O3

ENTRYPOINTS := \
	bnn-simple.c \

SOURCES := \
	bnn.c \

RESULTS_DIR=results
DATA_DIR=data
PLOTS_DIR=plots
BUILD_DIR=build
SRC_DIR=src
INCLUDE_DIR=include

INCLUDE_FLAGS=-I../../include  -I$(INCLUDE_DIR)
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

run: $(ENTRYPOINTS_FLAT) $(SRC_DIR)/generate_data.py
	@echo ...............................................................................
	@echo Running bnn-simple
	mkdir -p $(RESULTS_DIR)
	mkdir -p $(DATA_DIR)
	mkdir -p $(PLOTS_DIR)
	rm $(DATA_DIR)/*
	$(PYTHON) $(SRC_DIR)/generate_data.py $(N_DATA_PONTS) $(N_TEST_DATA_POINTS)
	$(PYTHON) $(SRC_DIR)/generate_random_values.py $(LEARNING_RATE) $(N) $(N_TEST_DATA_POINTS) $(N_REPETITIONS) $(N_DATA_PONTS) $(N_NODES) $(N_LAYERS)
	@echo Pascal
	./$(BUILD_DIR)/bnn-simple $(LEARNING_RATE) $(N) $(N_TEST_DATA_POINTS) $(N_REPETITIONS) $(N_DATA_PONTS) $(N_NODES) $(N_LAYERS) $(PRINT_FREQUENCY)
	@echo
	@echo PyTorch
	$(PYTHON) $(SRC_DIR)/experiment.py $(LEARNING_RATE) $(N) $(N_TEST_DATA_POINTS) $(N_REPETITIONS) $(N_DATA_PONTS) $(N_NODES) $(N_LAYERS) $(PRINT_FREQUENCY)
	$(PYTHON) $(SRC_DIR)/plot.py $(N_TEST_DATA_POINTS) $(N_REPETITIONS) $(N_DATA_PONTS)
	@echo ...............................................................................

clean:
	rm -fv $(BUILD_DIR)/*
