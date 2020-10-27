setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt -r requirements_dev.txt

lint:
	flake8

run:
	python src/main.py all -p default -n 1

run-test-all-algs:
	python src/main.py all -p test_all_algorithms -n 1 -v

build-synthetic-data:
	python scripts/synthetic_datasets.py

build-xor-data:
	python scripts/xor_dataset.py

download-cumida-data:
	python scripts/download_cumida_datasets.py
