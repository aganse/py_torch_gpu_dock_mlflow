.PHONY: env train run

env:
	bash make_env.bash

install:
	pip install -r requirements.txt

run:
	python project_driver.py

train:
	make env && make run

test:
	python -m pytest -q tests/
