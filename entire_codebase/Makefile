# SparkyFitness ML System — Unified Makefile
#
# Usage:
#   make setup-sparky    → first-time: init submodule + copy integration files (run once)
#   make run-all         → start complete system (ML + SparkyFitness)
#   make data            → run batch data pipeline
#   make train           → trigger retraining (async via API)
#   make train-direct    → run trainer container directly (blocking)
#   make promote         → promote Staging → Production
#   make rollback        → rollback to previous Production model
#   make smoke-test      → verify serving endpoint
#   make status          → show all container statuses

COMPOSE     = docker compose
SERVING_URL ?= http://localhost:8000
RETRAIN_URL ?= http://localhost:8080
MLFLOW_URL  ?= http://localhost:5000
SPARKY_URL  ?= http://localhost:3004

.PHONY: help setup setup-sparky build run-all run-infra run-serving run-monitoring \
        run-sparkyfitness data generate-data train train-direct retrain-status \
        promote rollback model-info smoke-test fairness drift-check \
        soda-checks status logs logs-serving logs-train stop clean \
        restart-serving pipeline mlflow-ui grafana-ui psql

help:
	@echo ""
	@echo "SparkyFitness ML System"
	@echo "======================"
	@echo ""
	@echo "  make setup-sparky    Init submodule + copy integration files (run once after clone)"
	@echo "  make run-all         Start complete system: ML pipeline + SparkyFitness UI"
	@echo "  make setup           Build images + start postgres + mlflow"
	@echo "  make run-all         Start the complete system"
	@echo "  make run-infra       Start postgres + mlflow only"
	@echo "  make run-serving     Start serving + retrain-api"
	@echo "  make run-monitoring  Start monitoring stack"
	@echo ""
	@echo "  make data            Run batch data pipeline"
	@echo "  make train           Trigger retraining via API (async)"
	@echo "  make train-direct    Run training container directly (blocking)"
	@echo "  make promote         Promote latest Staging → Production"
	@echo "  make rollback        Rollback to previous Production model"
	@echo ""
	@echo "  make smoke-test      Health check + sample prediction"
	@echo "  make fairness        Run fairness check on test.csv"
	@echo "  make status          All container statuses"
	@echo "  make stop / clean    Stop / destroy all"
	@echo ""

# ---------------------------------------------------------------------------
# Setup & Build
# ---------------------------------------------------------------------------
setup: build run-infra
	@echo "Waiting for infrastructure..."
	@sleep 10
	@echo "Ready. MLflow: $(MLFLOW_URL)"
	@echo "Next: make data && make train && make run-serving"

build:
	$(COMPOSE) build

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
run-infra:
	$(COMPOSE) up -d postgres mlflow
	@until docker exec sparky-postgres pg_isready -U sparky -d sparky >/dev/null 2>&1; do sleep 2; done
	@echo "PostgreSQL ready."
	@until curl -sf $(MLFLOW_URL)/health >/dev/null 2>&1; do sleep 3; done
	@echo "MLflow ready."

run-serving: run-infra
	$(COMPOSE) --profile runtime up -d serving retrain-api

run-monitoring: run-infra
	$(COMPOSE) --profile runtime up -d \
	  prometheus grafana postgres-exporter drift-monitor

run-sparkyfitness:
	$(COMPOSE) --profile runtime up -d sparkyfitness-db sparkyfitness-server sparkyfitness-frontend
	@echo "SparkyFitness running at $(SPARKY_URL)"

run-all:
	@# Ensure SparkyFitness submodule is initialised (safe to run on every boot)
	@if [ ! -f SparkyFitness/package.json ]; then \
	    echo "Initialising SparkyFitness submodule..."; \
	    git submodule update --init --recursive; \
	fi
	@# Create root .env from template when missing so compose can read variables
	@if [ ! -f .env ]; then \
	    echo "Creating .env from .env.example..."; \
	    cp .env.example .env; \
	fi
	$(COMPOSE) --profile pipeline up -d --build
	@echo ""
	@echo "System running:"
	@echo "  SparkyFitness: $(SPARKY_URL)"
	@echo "  Serving:      $(SERVING_URL)"
	@echo "  MLflow:       $(MLFLOW_URL)"
	@echo "  Grafana:      http://localhost:3000  (admin / sparky_admin)"
	@echo "  Prometheus:   http://localhost:9090"
	@echo "  Retrain API:  $(RETRAIN_URL)"

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
data:
	@echo "Running batch data pipeline..."
	$(COMPOSE) --profile pipeline run --rm batch-pipeline
	@echo "Training data compiled to /training-data volume."

generate-data:
	$(COMPOSE) --profile data up data-generator

# ---------------------------------------------------------------------------
# Training & Model Management
# ---------------------------------------------------------------------------
train: run-infra
	@echo "Triggering retraining..."
	curl -s -X POST "$(RETRAIN_URL)/trigger" \
	  -H "Content-Type: application/json" \
	  -d '{"reason":"manual","config":"configs/training/xgb_ranker.yaml","auto_promote":false}' \
	  | python3 -m json.tool
	@echo "Check progress: make retrain-status"

train-direct: run-infra
	$(COMPOSE) --profile pipeline run --rm trainer \
	  python -m src.training.retrain_pipeline \
	    --config configs/training/xgb_ranker.yaml \
	    --skip-data-checks

retrain-status:
	@curl -s "$(RETRAIN_URL)/status" | python3 -m json.tool

promote:
	@curl -s -X POST "$(RETRAIN_URL)/promote" \
	  -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
	@curl -s -X POST "$(SERVING_URL)/admin/reload" | python3 -m json.tool

rollback:
	@curl -s -X POST "$(RETRAIN_URL)/rollback" \
	  -H "Content-Type: application/json" -d '{}' | python3 -m json.tool
	@curl -s -X POST "$(SERVING_URL)/admin/reload" | python3 -m json.tool

model-info:
	@curl -s "$(SERVING_URL)/admin/model-info" | python3 -m json.tool

# ---------------------------------------------------------------------------
# Quality & Safeguarding
# ---------------------------------------------------------------------------
smoke-test:
	python3 scripts/smoke_test.py --url $(SERVING_URL)

fairness:
	python3 safeguarding/fairness_checker.py --test-csv output/test.csv

drift-check:
	$(COMPOSE) run --rm drift-monitor python src/data/drift_monitor.py --once

soda-checks:
	$(COMPOSE) --profile pipeline run --rm batch-pipeline \
	  python src/data/run_soda_checks.py \
	    --train-csv /output/train.csv \
	    --val-csv /output/val.csv \
	    --test-csv /output/test.csv

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
status:
	@docker ps --filter "name=sparky-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

logs:
	$(COMPOSE) logs -f --tail=50

logs-serving:
	docker logs -f sparky-serving

logs-train:
	docker logs -f sparky-trainer

stop:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v

restart-serving:
	docker restart sparky-serving

# ---------------------------------------------------------------------------
# Full pipeline (data → train → promote → smoke-test)
# ---------------------------------------------------------------------------
pipeline: data train-direct promote smoke-test
	@echo "Full pipeline complete."

# ---------------------------------------------------------------------------
# Dev helpers
# ---------------------------------------------------------------------------
mlflow-ui:
	@open $(MLFLOW_URL) 2>/dev/null || xdg-open $(MLFLOW_URL) 2>/dev/null || echo "Open: $(MLFLOW_URL)"

grafana-ui:
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Open: http://localhost:3000"

psql:
	docker exec -it sparky-postgres psql -U sparky -d sparky
