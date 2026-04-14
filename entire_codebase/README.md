# SparkyFitness — End-to-End ML System

Personalized recipe recommendation using XGBoost learning-to-rank.
A unified ML project: data → features → training → evaluation → registry → serving → monitoring → retraining.

---

## Repository Structure

```
.
├── src/                           ← All source code, organized by role
│   ├── training/                  ← Model training, registry, retraining API
│   │   ├── train.py               ← XGBoost LambdaRank training orchestrator
│   │   ├── ranking_data.py        ← Feature loading and string encoding
│   │   ├── metric.py              ← NDCG@10 evaluation
│   │   ├── mlflow_utils.py        ← Config reader + MLflow param logging
│   │   ├── ray_training.py        ← Distributed Ray training
│   │   ├── model_registry.py      ← MLflow Registry: promote, rollback, export
│   │   ├── retrain_pipeline.py    ← End-to-end: data checks → train → register
│   │   └── retrain_api.py         ← HTTP webhook API for retrain triggers
│   ├── serving/                   ← Inference API
│   │   ├── app_production.py      ← FastAPI: MLflow Registry load + hot-reload + logging
│   │   ├── model_loader.py        ← MLflow loader with local fallback
│   │   └── prediction_logger.py   ← Async predictions → PostgreSQL (feedback loop)
│   └── data/                      ← Data pipeline
│       ├── build_training_table.py← Feature engineering (cuisine, allergens, PCA)
│       ├── batch_pipeline.py      ← Merge prod interactions + compile training splits
│       ├── data_generator.py      ← Synthetic user interaction simulator
│       ├── drift_monitor.py       ← KS-test drift detection + retraining trigger
│       ├── ingest_to_object_store.py ← Upload raw data to Chameleon Swift
│       ├── online_feature_pipeline.py ← Real-time feature computation for serving
│       └── run_soda_checks.py     ← Soda Core data quality validation
│
├── configs/                       ← All configuration, organized by role
│   ├── training/
│   │   ├── xgb_ranker.yaml        ← Production training config (XGBoost LambdaRank)
│   │   ├── xgb_ranker_ray.yaml    ← Distributed Ray training config
│   │   └── baseline_popularity.yaml ← Simple popularity baseline
│   ├── serving/
│   │   └── serving.yaml           ← Serving config (port, poll interval, retention)
│   └── data/
│       ├── init_db.sql            ← PostgreSQL schema (users, recipes, interactions)
│       ├── init_feedback.sql      ← Feedback tables (prediction_log, drift_log, ...)
│       └── soda_checks.yml        ← Soda data quality rules
│
├── docker/                        ← All Dockerfiles in one place
│   ├── Dockerfile.training        ← Training + retraining API container
│   ├── Dockerfile.serving         ← FastAPI serving container
│   ├── Dockerfile.data            ← Data pipeline container
│   └── Dockerfile.mlflow          ← MLflow tracking server
│
├── requirements/                  ← Per-role Python dependencies
│   ├── training.txt
│   ├── serving.txt
│   └── data.txt
│
├── monitoring/                    ← Shared observability stack
│   ├── prometheus.yml             ← Scrape configs (serving, postgres, mlflow)
│   ├── alert_rules.yml            ← Alerts: latency, errors, drift, model staleness
│   ├── alertmanager.yml           ← Alert routing + auto-rollback webhook
│   └── grafana/
│       ├── dashboards/ml_system.json ← Pre-built ML system dashboard
│       └── provisioning/          ← Auto-provisioned datasources + dashboard pointers
│
├── safeguarding/
│   ├── SAFEGUARDING_PLAN.md       ← Full plan: fairness, privacy, explainability, ...
│   ├── fairness_checker.py        ← Group NDCG@10 gate + allergen safety check
│   └── explainability.py          ← SHAP per-prediction + global feature importance
│
├── contracts/                     ← Shared API contracts (input/output schema)
│   ├── recipe_ranker_input.sample.json
│   └── recipe_ranker_output.sample.json
│
├── scripts/                       ← Operational + dev utilities
│   ├── smoke_test.py              ← CI smoke test (health + prediction + latency SLA)
│   ├── benchmark.py               ← Load testing + latency profiling
│   └── convert_to_onnx.py         ← XGBoost → ONNX format conversion (optional)
│
├── data/                          ← Raw source data (Kaggle Food.com)
│   ├── RAW_recipes.csv
│   ├── RAW_interactions.csv
│   ├── PP_recipes.csv / PP_users.csv / ingr_map.pkl
│
├── output/                        ← Generated pipeline outputs (gitignored in prod)
│   ├── train.csv / val.csv / test.csv
│   ├── enriched_recipes.csv
│   └── manifest.json              ← SHA-256 hashes + git commit for reproducibility
│
├── docker-compose.yml             ← Single unified compose (all services, one context)
├── Makefile                       ← Runbook: setup, train, promote, rollback, ...
├── .env.example                   ← Environment variable template
├── .gitignore
└── .github/workflows/
    └── retrain.yml                ← CI/CD: quality gate → train → evaluate → promote
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      SparkyFitness ML System                         │
│                                                                      │
│  src/data/                src/training/          src/serving/        │
│  ┌────────────────┐       ┌──────────────┐       ┌──────────────┐   │
│  │ build_training_│       │  XGBoost     │       │  FastAPI     │   │
│  │ table.py       │──────▶│  LambdaRank  │──────▶│  /predict    │   │
│  │ batch_pipeline │       │  NDCG@10     │       │  /feedback   │   │
│  │ drift_monitor  │       │  Fairness    │       │  /explain    │   │
│  └────────────────┘       │  Gates       │       └──────┬───────┘   │
│         ▲                 └──────┬───────┘              │           │
│         │                       │ MLflow Registry       │           │
│         └───── feedback ─────── ┤ Staging → Production  │           │
│                loop             └───────────────────────┘           │
│                                                                      │
│  configs/training/*.yaml   configs/data/*.sql   configs/serving/     │
│  docker/Dockerfile.*       monitoring/           safeguarding/       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env          # fill in credentials

# 2. Build + start infrastructure
make setup                    # postgres + mlflow

# 3. Compile training data
make data

# 4. Train + register model
make train-direct

# 5. Start serving
make run-serving

# 6. Verify
make smoke-test

# 7. Start full monitoring stack
make run-monitoring

# --- Or, start everything at once ---
make run-all
```

---

## End-to-End Data Flow

```
1. RAW DATA  →  src/data/build_training_table.py
               Feature engineering: cuisine, allergens, nutrition, PCA history
               Output: configs/data/* SQL tables, output/{train,val,test}.csv

2. BATCH PIPELINE  →  src/data/batch_pipeline.py  (weekly + on drift)
               Merge PostgreSQL interactions + Soda quality checks
               Upload to Chameleon Swift (proj04-sparky-training-data)

3. TRAINING  →  src/training/retrain_pipeline.py
               Quality gates: NDCG@10 ≥ 0.55, fairness ±20%, allergen safety
               Register to MLflow Registry → Staging

4. PROMOTION  →  manual or scheduled auto-promote
               model_registry.promote_to_production()
               Serving hot-reloads the new Production version

5. SERVING  →  src/serving/app_production.py
               /predict → XGBoost inference → ranked recipes
               /feedback → user ratings → PostgreSQL
               /explain → SHAP feature attributions

6. MONITORING + LOOP
               drift_monitor.py: KS test every 5 min → POST /trigger on drift
               Prometheus alerts on degradation → auto-rollback
```

---

## Configuration Reference

| What | File |
|------|------|
| Training hyperparameters | `configs/training/xgb_ranker.yaml` |
| Quality gate thresholds | `.env` (`NDCG_THRESHOLD`, `IMPROVEMENT_THRESHOLD`) |
| Drift detection settings | `.env` (`DRIFT_THRESHOLD`, `CHECK_INTERVAL_SECONDS`) |
| Alert rules | `monitoring/alert_rules.yml` |
| Serving behaviour | `configs/serving/serving.yaml` |
| DB schema | `configs/data/init_db.sql` + `init_feedback.sql` |

---

## Team Responsibilities (3-person team)

| Role | Code | Config |
|------|------|--------|
| **Data** | `src/data/` | `configs/data/` |
| **Training** | `src/training/` | `configs/training/` |
| **Serving** | `src/serving/`, `monitoring/` | `configs/serving/` |

---

## Rollback

```bash
make rollback                            # rollback to previous Production model

# Or to a specific version:
curl -X POST http://localhost:8080/rollback \
  -H "Content-Type: application/json" -d '{"version":"3"}'
```

Automated rollback fires when Prometheus alerts `LowPredictionScores` or `HighErrorRate`.

---

## Safeguarding

See [safeguarding/SAFEGUARDING_PLAN.md](safeguarding/SAFEGUARDING_PLAN.md):
- **Fairness** — per-group NDCG@10 gate (±20%), allergen safety
- **Explainability** — SHAP attributions per prediction
- **Transparency** — model version + source in every response, full MLflow lineage
- **Privacy** — pseudonymized IDs, no PII in artifacts, 90-day feature retention
- **Accountability** — retraining log, GitHub environment approval gates, rollback trail
- **Robustness** — drift monitoring, auto-rollback, local model fallback
