VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.DEFAULT_GOAL := all

all: install-deps check-code prepare-data train-model mlflow-ui run-pipeline  

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-deps: venv
	$(PIP) install -r requirements.txt

format: install-deps
	$(VENV)/bin/black model_pipeline.py main.py

lint: install-deps
	$(VENV)/bin/pylint --fail-under=5.0 model_pipeline.py main.py

security-check: install-deps
	$(VENV)/bin/bandit -r model_pipeline.py main.py

check-code: format lint security-check

prepare-data: install-deps
	$(PYTHON) -c "from model_pipeline import prepare_data; prepare_data()"

train-model: prepare-data
	$(PYTHON) -c "from model_pipeline import prepare_data, train_model; X_train, y_train, _, _, _, _ = prepare_data(); model = train_model(X_train, y_train)"

mlflow-ui: install-deps
	mlflow ui --host 127.0.0.1 --port 5000 & \

run-pipeline: install-deps
	$(PYTHON) main.py

clean:
	rm -rf pycache
	rm -rf $(VENV)
	rm -f *.pkl

.PHONY: all install-deps check-code prepare-data train-model clean mlflow-ui run-pipeline
