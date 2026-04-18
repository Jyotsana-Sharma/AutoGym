# Project Reproducibility Guide

**Goal**: Clone the repo → run one command → entire system is live.

This guide covers two scenarios:
- **Scenario A** — ML system only (model training, serving, monitoring)
- **Scenario B** — Full stack (ML system + SparkyFitness app integrated)

---

## Prerequisites

Install these on your machine before starting:

| Tool | Minimum Version | Check |
|---|---|---|
| Docker | 24.0+ | `docker --version` |
| Docker Compose | v2.20+ (bundled with Docker Desktop) | `docker compose version` |
| Git | any | `git --version` |
| 8 GB RAM | — | Available to Docker |
| 10 GB disk | — | For images + volumes |

> Docker Desktop (Mac/Windows) includes Compose. On Linux: `sudo apt install docker-compose-plugin`.

No Python, Node, or any other runtime needed on your host machine — everything runs inside Docker.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/jyotsanasharma/AutoGym.git
cd AutoGym/entire_codebase
```

---

## Step 2 — Set Environment Variables (Optional)

All variables have safe defaults for local development. You only need to change them if:
- You want a custom database password
- You want to connect to Chameleon Cloud object storage (for dataset versioning)
- You want a custom Grafana password

```bash
# Copy the example env file
cp .env.example .env

# Edit only what you need (everything has defaults — this step is optional)
nano .env
```

**`.env` file reference**:

```bash
# PostgreSQL password (default: sparky_pass)
POSTGRES_PASSWORD=sparky_pass

# Grafana admin password (default: sparky_admin)
GRAFANA_PASSWORD=sparky_admin

# Chameleon Cloud Swift credentials (optional — only for cloud dataset upload)
# Leave empty to skip object storage and use local volumes instead
OS_AUTH_URL=
OS_APPLICATION_CREDENTIAL_ID=
OS_APPLICATION_CREDENTIAL_SECRET=
OS_REGION_NAME=CHI@TACC
```

If you skip this step entirely, the system uses all defaults and works out of the box.

---

## Scenario A — Full Bootstrap (Recommended for Graders)

This starts the complete integrated pipeline: PostgreSQL, MLflow, SparkyFitness, data compilation, one-shot training, serving, monitoring, and drift detection.

### One Command

```bash
docker compose --profile pipeline up -d --build
```

That's it. This single command:
1. Builds all Docker images from the single multi-stage `Dockerfile`
2. Starts PostgreSQL and MLflow, waits for them to be healthy
3. Applies the SparkyFitness recommendation route/UI integration
4. Runs one-shot data compilation and training/evaluation/registration jobs
5. Starts the serving API, which loads the Production model from MLflow Registry
6. Starts the monitoring stack (Prometheus + Grafana alerting)
7. Starts the drift monitor (checks for feature drift every 5 minutes)

After the first successful bootstrap, normal production startup should use:

```bash
docker compose --profile runtime up -d
```

### Wait for Startup (~2–3 minutes on first run)

```bash
# Watch all containers come up
docker compose logs -f

# Or check status with:
docker ps --filter "name=sparky-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

All containers should show `healthy` or `Up` within 3 minutes.

### Verify It Works

```bash
# 1. Check serving API is alive
curl http://localhost:8000/health

# Expected:
# {"status":"healthy","model":"xgb_recipe_ranker","model_version":"..."}

# 2. Run a sample prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "model_name": "xgb_ranker",
    "instances": [{
      "user_id": 1,
      "recipe_id": 33096,
      "minutes": 30,
      "cuisine": "italian",
      "n_ingredients": 8,
      "n_steps": 5,
      "avg_rating": 4.2,
      "n_reviews": 85,
      "calories": 420.0,
      "protein": 25.0,
      "total_fat": 15.0,
      "carbohydrate": 50.0,
      "sugar": 6.0,
      "sodium": 580.0,
      "saturated_fat": 4.5,
      "has_milk": 1,
      "has_egg": 0,
      "has_nuts": 0,
      "has_peanut": 0,
      "has_fish": 0,
      "has_shellfish": 0,
      "has_wheat": 1,
      "has_soy": 0,
      "has_sesame": 0,
      "daily_calorie_target": 2000.0,
      "protein_target_g": 50.0,
      "carbs_target_g": 250.0,
      "fat_target_g": 65.0,
      "user_vegetarian": 0,
      "user_vegan": 0,
      "user_gluten_free": 0,
      "user_dairy_free": 0,
      "user_low_sodium": 0,
      "user_low_fat": 0,
      "history_pc1": 0.3,
      "history_pc2": -0.1,
      "history_pc3": 0.05
    }]
  }' | python3 -m json.tool

# 3. Check SHAP explainability endpoint
curl -s -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "instance": {
      "user_id": 1, "recipe_id": 33096, "calories": 420.0,
      "protein": 25.0, "minutes": 30, "avg_rating": 4.2
    },
    "top_k": 5
  }' | python3 -m json.tool
```

### Access the UIs

| Interface | URL | Credentials |
|---|---|---|
| MLflow (experiments + model registry) | http://localhost:5000 | none |
| Grafana (monitoring dashboards) | http://localhost:3000 | admin / sparky_admin |
| Prometheus (raw metrics) | http://localhost:9090 | none |
| Serving API docs (Swagger) | http://localhost:8000/docs | none |
| Retrain API | http://localhost:8080 | none |

---

## Scenario B — Full Stack with SparkyFitness Integration

Everything needed is already in this repo:
- `sparkyfitness-integration/` — the ML recommendation patch files for SparkyFitness
- `docker-compose.yml` — SparkyFitness containers (`sparkyfitness-db`, `sparkyfitness-server`, `sparkyfitness-frontend`) already wired with `ML_RECOMMENDATION_URL=http://sparky-serving:8000`
- `scripts/setup-sparkyfitness.sh` — applies all patches automatically
- `.env.sparky.example` — all environment variables with safe defaults

### Two commands to run everything

**Step B1 — Initialise the submodule** (once, after first clone):

```bash
git submodule update --init --recursive
```

> If you cloned with `--recurse-submodules` this is already done — skip it.

**Step B2 — Start everything**:

```bash
docker compose --profile pipeline up -d
```

The `sparkyfitness-setup` container runs automatically first — no manual script needed. It runs `sparkyfitness-integration/apply_integration.py`, copies all integration patch files into the SparkyFitness source, patches `SparkyFitnessServer.ts` to register `/api/recommendations`, patches `Foods.tsx` to render `RecipeRecommendations`, and creates `SparkyFitness/.env`. Only after it exits successfully do `sparkyfitness-server` and `sparkyfitness-frontend` start.

This starts 14 services in the correct order. `batch-pipeline`, `trainer`, and
`sparkyfitness-setup` run once and exit; the app, serving, retraining, database,
and monitoring services stay running at steady state.

After the first successful bootstrap, you can restart or operate only the
steady-state system with the smaller runtime profile:

```bash
docker compose --profile runtime up -d
```

The runtime profile starts 11 services and skips the one-shot containers
`sparkyfitness-setup`, `batch-pipeline`, and `trainer`. Use the full `pipeline`
profile again whenever you intentionally want to rebuild training data and
train/register a fresh model from scratch. Normal runtime startup does not
retrain; serving loads the current Production model from MLflow Registry.

```
postgres + mlflow
      ↓
batch-pipeline (compile training data)        sparkyfitness-setup (copy patches, create .env)
      ↓  exits 0                                      ↓  exits 0
trainer (XGBoost + fairness gate + MLflow)    sparkyfitness-db (healthy)
      ↓  exits 0                                      ↓
sparky-serving  ←─────────────────────────┐  sparkyfitness-server ──────────────────────┘
retrain-api                               │  sparkyfitness-frontend (port 3004)
drift-monitor                             │  ML_RECOMMENDATION_URL=http://sparky-serving:8000
prometheus + grafana alerting             │  (both on sparky-net — talk by container name)
          └──────────────────────────────────────────────────────────────────────────────────┘
```

### Verify

```bash
# ML system healthy
curl http://localhost:8000/health

# SparkyFitness UI
open http://localhost:3004/foods
# Navigate to Foods → Recommended Recipes
# You should see ML-ranked recipe cards
```

### Access the UIs

| Interface | URL |
|---|---|
| SparkyFitness App | http://localhost:3004 |
| ML Serving API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin / sparky_admin) |

If the ML service is unreachable, SparkyFitness automatically falls back to a protein-proximity heuristic — the Foods page still works.

---

## Running the Training Pipeline

The system ships with a pre-trained model. To run the full training pipeline yourself:

### Option 1 — Direct training (blocking, shows all output)

```bash
docker compose run --rm trainer
```

This runs the full pipeline:
1. Loads training data from the shared volume
2. Trains XGBoost LambdaRank (NDCG@10 objective)
3. Runs fairness gate (per-group NDCG@10 must stay within 20% of overall NDCG@10, with allergen safety <1%)
4. Generates SHAP global feature importance
5. Registers model to MLflow Registry if all gates pass
6. Serving API hot-reloads the new model within 60 seconds

### Option 2 — Trigger via API (async)

```bash
curl -X POST http://localhost:8080/trigger \
  -H "Content-Type: application/json" \
  -d '{"reason":"manual","config":"configs/training/xgb_ranker.yaml","auto_promote":false}'
```

### Option 3 — Using make (if make is available)

```bash
make train-direct   # blocking, direct
make train          # async via retrain-api
make promote        # promote Staging → Production after training
make smoke-test     # verify serving is healthy
```

---

## Generating Fresh Training Data

If you want to populate the database with fresh synthetic user interactions before training:

```bash
# Run batch pipeline to compile training CSVs
docker compose run --rm batch-pipeline

# Then train on the fresh data
docker compose run --rm trainer
```

---

## Full Pipeline in One Command (Data → Train → Serve → Monitor)

```bash
docker compose --profile pipeline up -d
```

That's it. Docker Compose executes the stages in strict order using `depends_on` with `condition: service_completed_successfully`:

```

For normal steady-state restarts after bootstrap:

```bash
docker compose --profile runtime up -d
```
postgres + mlflow start and pass healthchecks
        ↓
batch-pipeline runs (compiles train/val/test CSVs)
        ↓  exits 0
trainer runs (XGBoost + fairness gate + MLflow registration)
        ↓  exits 0
serving starts (loads model, serves /predict /explain)
retrain-api starts
drift-monitor starts (checks for feature drift every 5 min)
prometheus + grafana start
sparkyfitness-setup applies app integration, then SparkyFitness starts
```

If **any step fails** (e.g. fairness gate rejects the model), the chain stops — serving never starts. Fix the issue and re-run the same command.

**Watch progress in real time:**
```bash
docker compose --profile pipeline logs -f
```

**Verify when done:**
```bash
curl http://localhost:8000/health
```

---

## Stopping and Cleaning Up

```bash
# Stop all containers (keeps volumes/data)
docker compose down

# Stop and delete all data (full reset)
docker compose down -v

# Remove all built images too
docker compose down -v --rmi all
```

---

## Troubleshooting

### "Cannot connect to the Docker daemon"
Start Docker Desktop (Mac/Windows) or run `sudo systemctl start docker` (Linux).

### Containers exit immediately
```bash
# Check logs for the failing container
docker compose logs <service-name>
# e.g.: docker compose logs serving
```

### Port already in use
Another process is using a port. Either stop the conflicting process or override the port:
```bash
# Override serving port to 8001
SERVING_PORT=8001 docker compose --profile runtime up -d serving
```
Or edit `docker-compose.yml` and change the left side of the port mapping (e.g., `"8001:8000"`).

### MLflow healthcheck fails / serving won't start
```bash
# Check MLflow is running
curl http://localhost:5000/health

# If not, restart it
docker compose restart mlflow

# Force-recreate if still broken
docker compose up -d --force-recreate mlflow
```

### Model not loading (serving returns 503)
```bash
# Check serving logs
docker logs sparky-serving --tail=50

# The serving container loads the model from MLflow on startup.
# If no model is registered yet, run training first:
docker compose run --rm trainer
```

### PostgreSQL connection refused
```bash
# Wait for postgres to finish initializing (can take 30–60 seconds first run)
docker exec sparky-postgres pg_isready -U sparky -d sparky

# If failing, check logs:
docker logs sparky-postgres --tail=30
```

### "No space left on device" during build
Docker image cache is full.
```bash
docker system prune -a --volumes
# Then re-run docker compose build
```

---

## What Each Container Does

| Container | Port | What it does |
|---|---|---|
| `sparky-postgres` | 5433 | Stores user interactions, drift logs, predictions |
| `sparky-mlflow` | 5000 | Tracks experiments, stores model artifacts, manages registry |
| `sparky-serving` | 8000 | Serves `/predict`, `/explain`, `/health`, `/metrics` |
| `sparky-retrain-api` | 8080 | Receives retraining webhook triggers |
| `sparky-data-generator` | — | Continuously generates synthetic user interactions |
| `sparky-batch-pipeline` | — | Compiles train/val/test CSVs (runs once then exits) |
| `sparky-trainer` | — | Trains XGBoost model (runs once then exits) |
| `sparky-drift-monitor` | — | KS-test on 19 features every 5 minutes |
| `sparky-prometheus` | 9090 | Scrapes and stores metrics |
| `sparky-grafana` | 3000 | Dashboards for latency, drift, training history |
| `sparky-postgres-exporter` | 9187 | Exports PostgreSQL metrics to Prometheus |
| `sparkyfitness-setup` | — | Applies the recommendation route/UI integration (runs once then exits) |
| `sparkyfitness-db` | — | SparkyFitness application database |
| `sparkyfitness-server` | 3010 | SparkyFitness API with recommendation bridge |
| `sparkyfitness-frontend` | 3004 | SparkyFitness UI with recommendation cards |

---

## Environment Variable Reference

| Variable | Default | Used By | Description |
|---|---|---|---|
| `POSTGRES_PASSWORD` | `sparky_pass` | postgres, all services | Database password |
| `GRAFANA_PASSWORD` | `sparky_admin` | grafana | Grafana admin password |
| `ROLLBACK_WEBHOOK_TOKEN` | _(empty)_ | grafana, retrain-api | Bearer token required before Grafana alert rollback is enabled |
| `ROLLBACK_ALERT_NAMES` | `HighErrorRate` | retrain-api | Comma-separated critical alert names allowed to trigger rollback |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | trainer, serving | MLflow server address |
| `NDCG_THRESHOLD` | `0.55` | trainer | Minimum NDCG@10 to register model |
| `DRIFT_THRESHOLD` | `0.05` | drift-monitor | KS-test p-value threshold |
| `CHECK_INTERVAL_SECONDS` | `300` | drift-monitor | Drift check frequency (seconds) |
| `LOG_PREDICTIONS` | `true` | serving | Enable inference feature logging |
| `MODEL_FALLBACK_PATH` | `/models/xgboost_ranker.json` | serving | Fallback if MLflow unavailable |
| `OS_AUTH_URL` | _(empty)_ | batch-pipeline | Chameleon Cloud auth (optional) |
| `OS_APPLICATION_CREDENTIAL_ID` | _(empty)_ | batch-pipeline | Swift credential ID (optional) |
| `OS_APPLICATION_CREDENTIAL_SECRET` | _(empty)_ | batch-pipeline | Swift credential secret (optional) |

---

## Tested On

| Platform | Status |
|---|---|
| macOS 14 (Apple Silicon, Docker Desktop 4.x) | Verified |
| macOS 13 (Intel, Docker Desktop 4.x) | Verified |
| Ubuntu 22.04 (Docker CE + Compose plugin) | Verified |
| Chameleon Cloud KVM@TACC (Ubuntu 22.04) | Deployed (production) |
| Windows 11 (WSL2 + Docker Desktop) | Should work — not explicitly tested |
