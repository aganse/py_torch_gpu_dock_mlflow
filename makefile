.PHONY: env train run

VERSION := $(strip $(shell cat VERSION))

env:
	bash make_env.bash

install:
	pip install -r requirements.txt

run:
	python train.py

train:
	make env && make run

unittest:
	python -m pytest -q tests/

run_mlproject:
	./project_driver.bash

build:
	@echo "Building torch-gpu-mlflow image (version $(VERSION))"
	docker build -t torch-gpu-mlflow:$(VERSION) -t torch-gpu-mlflow:latest .
	# docker run --rm torch-gpu-mlflow --epochs 1 --batch_size 
