.PHONY: env train run

VERSION := $(strip $(shell cat VERSION))

build:
	@echo "Building torch-gpu-mlflow image (version $(VERSION))"
	docker build -t torch-gpu-mlflow:$(VERSION) -t torch-gpu-mlflow:latest .
	# docker run --rm torch-gpu-mlflow --epochs 1 --batch_size 

run_mlproject:
	./project_driver.bash

unittest:
	python -m pytest -q tests/

dev_install:
	pip install -r requirements.txt

dev_env:
	bash make_env.bash

dev_train:
	python train.py
