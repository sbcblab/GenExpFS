setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt -r requirements_dev.txt

lint:
	flake8

run:
	python src/main.py

build-synthetic-data:
	python scripts/synthetic_datasets.py
