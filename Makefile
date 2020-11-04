setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt -r requirements_dev.txt

lint:
	flake8

run:
	python src/main.py all -p default -n 1 -vv

run-test-algorithms:
	python src/main.py all -p test_algorithms -n 1 -vv

run-test-pipeline:
	python src/main.py all -p test_pipeline -n 1 -vv

run-reduced:
	python src/main.py all -p reduced -n 31 -vv

build-synthetic-data:
	python scripts/synthetic_datasets.py

build-xor-data:
	python scripts/xor_dataset.py

download-cumida-data:
	python scripts/download_cumida_datasets.py
