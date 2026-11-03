SWIFT ?= swift
DATA ?= .tmp
EPOCHS ?= 35
WEIGHTS ?= .tmp/weights.safetensors
RESOURCES ?= .tmp/resources
PUBLIC ?= .tmp/public
PORT ?= 8080

.PHONY: all build test train eval react server clean help

all: build test

build:
	$(SWIFT) build -c release

test:
	$(SWIFT) test

train:
	$(SWIFT) run -c release train -d $(DATA) -e $(EPOCHS) --export-weights $(WEIGHTS)

eval:
	$(SWIFT) run -c release train -d $(DATA) -e 0 --import-weights $(WEIGHTS)

react:
	$(SWIFT) run -c release react -d $(DATA) -w $(WEIGHTS) --resources $(RESOURCES)

server:
	$(SWIFT) run -c release server --port $(PORT) --weights $(WEIGHTS) --resources $(RESOURCES)

clean:
	$(SWIFT) package clean

help:
	@echo "make build        Build all targets in release mode"
	@echo "make test         Run unit test suite"
	@echo "make train        Train Hawk model on $(DATA) -> $(WEIGHTS)"
	@echo "make eval         Evaluate trained weights on $(DATA)"
	@echo "make react        Run CLI reaction stream on $(DATA)"
	@echo "make server       Start WebSocket simulation server on port $(PORT)"
	@echo "make clean        Clean Swift build artifacts"
