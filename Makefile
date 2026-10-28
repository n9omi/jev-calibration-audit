CONFIG ?= configs/experiment.yaml

.PHONY: install demo test lint fetch build run-jev run-llm run-baselines evaluate all

install:
	pip install -e ".[jev,llm,dev]"

demo:
	jev-audit demo

test:
	pytest

lint:
	ruff check src tests && ruff format --check src tests

fetch:
	jev-audit fetch -c $(CONFIG)

build:
	jev-audit build -c $(CONFIG)

run-jev:
	jev-audit run -c $(CONFIG) --arm jev

run-llm:
	jev-audit run -c $(CONFIG) --arm llm_sonnet

run-baselines:
	jev-audit run -c $(CONFIG) --arm rules
	jev-audit run -c $(CONFIG) --arm climatology

evaluate:
	jev-audit evaluate -c $(CONFIG)

all:
	jev-audit all -c $(CONFIG)
