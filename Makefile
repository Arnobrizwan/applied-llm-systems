.PHONY: help test demo lint clean install

help:
	@echo "make install  - install dev dependencies (pytest only)"
	@echo "make test     - run every test in the repo"
	@echo "make demo     - run all 15 project demos end to end"
	@echo "make clean    - remove caches and generated artefacts"

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest

demo:
	@python3 scripts/run_all_demos.py

lint:
	python3 -m compileall -q llmkit projects scripts && echo "syntax ok"

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .pytest_cache -type d -prune -exec rm -rf {} +
	rm -rf artifacts
