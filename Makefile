.PHONY: setup install lint test test-unit test-integration generate-data train-gnn train-sim-swap \
        train-federated serve serve-dev docker-up docker-down clean

# ── Environment ───────────────────────────────────────────────────────────────
setup:
	pip install uv
	uv pip install -e ".[dev]"
	pre-commit install

install:
	uv pip install -e ".[dev,gnn]"

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check .
	ruff format --check .
	mypy shared/ data_generation/ models/ api/

format:
	ruff format .
	ruff check --fix .

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --timeout=60

test-e2e:
	pytest tests/e2e/ -v --timeout=120

# ── Data ──────────────────────────────────────────────────────────────────────
generate-data:
	python -m data_generation.scripts.generate_training_data \
		--n-accounts 50000 \
		--fraud-rate 0.023 \
		--output-dir data/

validate-data:
	python -m data_generation.scripts.validate_dataset --data-dir data/

# ── Training ──────────────────────────────────────────────────────────────────
train-sim-swap:
	python -m models.sim_swap.train \
		--data-path data/processed/sim_swap_features.parquet \
		--experiment-name sim-swap-detector

train-gnn:
	python -m models.gnn.train \
		--graph-path data/graphs/hetero_graph.pt \
		--experiment-name gnn-fraud-ring

train-federated:
	python -m models.federated.simulate \
		--data-path data/raw/transactions.parquet \
		--num-rounds 50

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

serve-dev:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

stream:
	faust -A streaming.app worker -l info

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Services ready. MLflow UI: http://localhost:5001  Grafana: http://localhost:3000"

docker-down:
	docker-compose down -v

docker-build:
	docker-compose build

# ── Utilities ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	rm -rf data/raw data/processed data/graphs mlruns

mlflow-ui:
	mlflow ui --port 5001

kafka-topics:
	docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list
