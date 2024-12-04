CC=clang

OPTIMISATION_LEVEL=-O3

ENTRYPOINTS := \
	nn-simple\

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

SOURCES_FLAT:= $(SOURCES:$(SRC_DIR)/%=%)
OBJECTS := $(SOURCES_FLAT:%=$(BUILD_DIR)/%.o)
LIB := $(OBJECTS:$(BUILD_DIR)/main.c.o%=%)

ENTRYPOINTS_FLAT := $(ENTRYPOINTS:%=$(BUILD_DIR)/%)

$(BUILD_DIR)/%.c.o: $(SRC_DIR)/%.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

.PHONY := \
	run \
	clean \

.SECONDEXPANSION:
$(addsuffix .c.o,$(ENTRYPOINTS_FLAT)):: $$(patsubst $(BUILD_DIR)/%.c.o,$(SRC_DIR)/%.c,$$@)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $<  $(INCLUDE_FLAGS) -o $@

.SECONDEXPANSION:
$(ENTRYPOINTS_FLAT): $$@.c.o $(LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(FRAMEWORK_FLAGS)

run: $(ENTRYPOINTS_FLAT)  $(SRC_DIR)/generate_data.py
	@echo ...............................................................................
	@echo Running nn-simple
	mkdir -p $(RESULTS_DIR)
	mkdir -p $(DATA_DIR)
	mkdir -p $(PLOTS_DIR)
	python $(SRC_DIR)/generate_data.py
	$(foreach executable,$(ENTRYPOINTS_FLAT),$(executable);)
	@echo ...............................................................................

clean:
	rm -fv $(BUILD_DIR)/*
