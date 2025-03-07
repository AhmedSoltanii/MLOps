VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

IMAGE_NAME = ahmedsoltani/4ds2_mlops
TAG = latest
CONTAINER_NAME = mlops_container

.DEFAULT_GOAL := all

all: install-deps check-code prepare-data train-model mlflow-ui run-pipeline  

# Virtual environment setup
venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-deps: venv
	$(PIP) install -r requirements.txt

# Code quality checks
format: install-deps
	$(VENV)/bin/black model_pipeline.py main.py

lint: install-deps
	$(VENV)/bin/pylint --fail-under=5.0 model_pipeline.py main.py

security-check: install-deps
	$(VENV)/bin/bandit -r model_pipeline.py main.py

check-code: format lint security-check

# Data preparation and model training
prepare-data: install-deps
	$(PYTHON) -c "from model_pipeline import prepare_data; prepare_data()"

train-model: prepare-data
	$(PYTHON) -c "from model_pipeline import prepare_data, train_model; X_train, y_train, _, _, _, _ = prepare_data(); model = train_model(X_train, y_train)"

# MLflow UI
mlflow-ui: install-deps
	mlflow ui --host 127.0.0.1 --port 5000 &

run-pipeline: install-deps
	$(PYTHON) main.py

# Docker tasks
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

docker-push: docker-build
	docker push $(IMAGE_NAME):$(TAG)

docker-run:
	docker run -d -p 8000:8000 --name $(CONTAINER_NAME) $(IMAGE_NAME):$(TAG)

docker-stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

clean:
	rm -rf pycache
	rm -rf $(VENV)
	rm -f *.pkl
	docker rmi $(IMAGE_NAME):$(TAG) || true

.PHONY: all install-deps check-code prepare-data train-model clean mlflow-ui run-pipeline docker-build docker-push docker-run docker-stop
