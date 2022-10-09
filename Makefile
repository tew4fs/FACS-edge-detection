.DEFAULT_GOAL := build

run:
	python facs/main.py

lint:
	flake8 .

format:
	black .

test:
	pytest

build: format lint test run