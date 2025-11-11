.PHONY: env train run

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
	docker build -t torch-gpu-mlflow .
	# docker run --rm torch-gpu-mlflow --epochs 1 --batch_size 
