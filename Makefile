.PHONY: install build publish publish-test clean test

PYTHON := $(shell command -v python3 || command -v python)

install:
	pip3 install -e ".[dev]"

build:
	pip3 install --quiet build twine
	$(PYTHON) -m build

publish-test: build
	twine upload --repository testpypi dist/*

publish: build
	twine upload dist/*

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf dist/ build/ *.egg-info __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
