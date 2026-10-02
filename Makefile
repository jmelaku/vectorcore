VENV := .venv
PYTHON := $(VENV)/bin/python
CMAKE := $(VENV)/bin/cmake
NINJA := $(VENV)/bin/ninja
BUILD := build
PYBIND11_DIR := $(shell $(PYTHON) -m pybind11 --cmakedir 2>/dev/null)

.PHONY: setup configure build test benchmark example clean

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

configure:
	$(CMAKE) -S . -B $(BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_MAKE_PROGRAM=$(abspath $(NINJA)) \
		-DPython_EXECUTABLE=$(abspath $(PYTHON)) -Dpybind11_DIR=$(PYBIND11_DIR)

build: configure
	$(CMAKE) --build $(BUILD) --parallel

test: build
	$(CMAKE) --build $(BUILD) --target test

benchmark: build
	PYTHONPATH=$(BUILD) $(PYTHON) benchmarks/benchmark.py

example: build
	$(BUILD)/linear_regression
	PYTHONPATH=$(BUILD) $(PYTHON) examples/two_layer.py

clean:
	$(CMAKE) -E remove_directory $(BUILD)
