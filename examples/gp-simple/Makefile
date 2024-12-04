CC=clang

OPTIMISATION_LEVEL=-O0

ENTRYPOINT=example

RESULTS_DIR=results
DATA_DIR=data
PLOTS_DIR=plots
BUILD_DIR=build

INCLUDE_FLAGS=-I../../include
CFLAGS+=$(OPTIMISATION_LEVEL)
LDFLAGS=-L../../libs/ -lpascal
FRAMEWORK_FLAGS=

.PHONY := \
	run \
	clean \

$(BUILD_DIR)/$(ENTRYPOINT).c.o: $(ENTRYPOINT).c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $<  $(INCLUDE_FLAGS) -o $@

$(BUILD_DIR)/$(ENTRYPOINT): $(BUILD_DIR)/$(ENTRYPOINT).c.o
	$(CC) -o $@ $^ $(LDFLAGS) $(FRAMEWORK_FLAGS)

run: $(BUILD_DIR)/$(ENTRYPOINT) $(ENTRYPOINT).c plot.py
	@echo ...............................................................................
	@echo Running gp-simple
	mkdir -p $(RESULTS_DIR)
	mkdir -p $(PLOTS_DIR)
	./$(BUILD_DIR)/$(ENTRYPOINT); python plot.py
	@echo ...............................................................................

clean:
	rm -fv $(BUILD_DIR)/*
