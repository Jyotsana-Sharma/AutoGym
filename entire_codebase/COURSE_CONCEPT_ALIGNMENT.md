# Course Concept Alignment

This document maps the integrated SparkyFitness ML recommendation system to the
major ML Systems Engineering and Operations concepts covered in the course
notes and labs.

References:

- Course notes landing page: <https://ffund.github.io/ml-sys-ops-notes/>
- Course lab/topic overview: <https://ffund.github.io/ml-sys-ops/docs/instructor>

The instructor overview lists the relevant hands-on areas used below:
Chameleon cloud computing, MLOps pipeline, persistent storage, MLflow
experiment tracking, Ray training, model serving, offline/online evaluation,
and closing the feedback loop.

---

## Concept Coverage Matrix

| Course concept | How this project uses it | Main files |
|---|---|---|
| ML system design | The ML feature is not a standalone notebook; it is integrated into the SparkyFitness Foods user flow, with React → Express → FastAPI → XGBoost → feedback/logging. | `sparkyfitness-integration/`, `SparkyFitness/`, `src/serving/app_production.py` |
| Chameleon cloud deployment | Deployment guide targets KVM@TACC for runtime and CHI@TACC Swift for object storage. | `CHAMELEON_DEPLOYMENT.md`, `docker-compose.yml` |
| Reproducible infrastructure | Single multi-stage `Dockerfile`, Docker Compose profiles, Makefile runbook, explicit environment templates. | `Dockerfile`, `docker-compose.yml`, `Makefile`, `.env.example` |
| Persistent storage | Uses Docker volumes for Postgres, MLflow DB/artifacts, model cache, Grafana, Prometheus, and SparkyFitness application state. Uses Swift buckets for raw/versioned data. | `docker-compose.yml`, `src/data/ingest_to_object_store.py`, `src/data/batch_pipeline.py` |
| Data ingestion | Raw Food.com data can be pulled from Swift or local data volume; ingestion validates expected files and schema. | `src/data/ingest_to_object_store.py`, `src/data/batch_pipeline.py` |
| Data quality at ingestion | Soda checks plus runtime ingestion checks block bad data before training. | `src/data/run_soda_checks.py`, `configs/data/soda_checks.yml`, `src/data/drift_monitor.py` |
| Training set compilation | Batch pipeline compiles train/val/test data, merges production interactions, validates label balance and temporal integrity, and writes a manifest. | `src/data/batch_pipeline.py`, `src/data/build_training_table.py`, `output/manifest.json` |
| Data/version lineage | Dataset manifest records split sizes, hashes, dates, and code/version metadata; MLflow logs config and artifacts. | `output/manifest.json`, `src/training/train.py`, `src/training/mlflow_utils.py` |
| Model training infrastructure | XGBoost LambdaRank is the production model; Ray config exists for distributed training. | `src/training/train.py`, `src/training/ray_training.py`, `configs/training/xgb_ranker*.yaml` |
| Experiment tracking | Every run logs metrics, parameters, datasets, configs, artifacts, fairness, and explainability to MLflow. | `src/training/train.py`, `src/training/model_registry.py`, `src/training/retrain_pipeline.py` |
| Model registry | Models are registered only after gates pass. Serving fetches the current Production model from MLflow Registry; runtime startup does not retrain. | `src/training/model_registry.py`, `src/serving/model_loader.py`, `src/serving/app_production.py` |
| CI/CD and automation | GitHub Actions performs data quality, training/evaluation/registration, optional promotion, and rollback on failed smoke test. Runtime retraining can be triggered by drift/manual/schedule. | `.github/workflows/retrain.yml`, `src/training/retrain_api.py` |
| Deployment profiles | `pipeline` bootstraps data/training/app once; `runtime` starts 11 long-running containers and loads model from registry. | `docker-compose.yml`, `project_reproducibility.md` |
| Model serving | FastAPI `/predict`, `/feedback`, `/explain`, `/health`, `/metrics`; hot-reloads Production model from registry. | `src/serving/app_production.py`, `src/serving/model_loader.py` |
| Serving fallback | If MLflow Registry is unavailable, serving falls back to local exported model file. SparkyFitness app also falls back to a protein-proximity heuristic if ML serving is unavailable. | `src/serving/model_loader.py`, `sparkyfitness-integration/SparkyFitnessServer/services/recommendationService.ts` |
| Offline evaluation | Training evaluates NDCG@10 on held-out test data and compares against baseline. | `src/training/metric.py`, `src/training/train.py` |
| Model quality gates | Registration requires NDCG threshold, limited regression versus current Production, and fairness/allergen safety. | `src/training/model_registry.py`, `src/training/retrain_pipeline.py` |
| Online evaluation and monitoring | Prometheus/Grafana monitor serving latency, errors, score distribution, prediction volume, model version, retraining status, and DB metrics. | `src/serving/app_production.py`, `src/training/retrain_api.py`, `monitoring/` |
| Drift monitoring | KS-test compares live inference features against training baseline every 300 seconds; drift triggers retraining API. | `src/data/drift_monitor.py` |
| Closing the feedback loop | Serving logs predictions and features to ML Postgres; SparkyFitness forwards UI feedback to ML `/feedback`; shared `request_id` and `recommendation_id` join app feedback with ML logs. | `src/serving/prediction_logger.py`, `sparkyfitness-integration/SparkyFitnessServer/services/recommendationService.ts`, `src/data/batch_pipeline.py` |
| Human-in-the-loop deployment control | Models register to Staging; promotion to Production can be manual/CI gated. CI rollback and authenticated Grafana rollback restore previous archived Production versions. | `src/training/model_registry.py`, `src/training/retrain_api.py`, `.github/workflows/retrain.yml` |
| Safeguarding | Fairness gate, allergen safety, SHAP explanations, transparent model metadata, pseudonymized IDs, retention cleanup, rollback trail, model card, and dataset card. | `safeguarding/`, `SAFEGUARDING_PLAN.md`, `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md` |
| Operational observability | Grafana dashboards, provisioned Grafana alerting, and Prometheus metrics support production monitoring without a separate Alertmanager container. | `monitoring/grafana/`, `monitoring/prometheus.yml`, `monitoring/alert_rules.yml` |
| Train/serve contract testing | Serving feature columns are centralized and tested against training feature inference so metadata IDs do not become model inputs. | `src/serving/feature_contract.py`, `src/training/ranking_data.py`, `tests/test_feature_contract.py` |
| Team role integration | Data, training, and serving responsibilities share one Compose stack, one MLflow, one monitoring stack, and one app integration path. | `MILESTONE_DELIVERABLES.md`, `docker-compose.yml` |

---

## Runtime Behavior

The system intentionally separates bootstrap/retraining from steady-state
runtime:

```bash
# First-time bootstrap or intentional full retrain:
docker compose --profile pipeline up -d --build

# Normal production runtime after a model exists in MLflow Registry:
docker compose --profile runtime up -d
```

`pipeline` includes the one-shot setup/data/training jobs. `runtime` starts only
the 11 long-running containers. In runtime mode, the model is not retrained on
startup; `sparky-serving` loads the current Production model from MLflow
Registry and polls for newer Production versions.

---

## Extension Points Implemented

The previously listed course-aligned extension points are now implemented:

| Extension | Implementation |
|---|---|
| Shared request/recommendation identifier across SparkyFitness and ML logs | SparkyFitness generates a `recommendation_id` per candidate and request-level `request_id`; both are stored in app cache/interactions and ML `prediction_log`, `user_feedback`, and `inference_features`. |
| Grafana alert provisioning | `monitoring/grafana/provisioning/alerting/sparky-alerts.yml` provisions a critical `HighErrorRate` Grafana rule and rollback contact point. |
| Direct alert-triggered rollback webhook | `POST /alerts/rollback` in `retrain_api.py` requires a bearer token and only rolls back for configured critical alert names. |
| Stronger train/serve feature contract tests | `tests/test_feature_contract.py` verifies training inference matches the serving feature contract and excludes request metadata. |
| Model card / dataset card | `docs/MODEL_CARD.md` and `docs/DATASET_CARD.md` document intended use, data, safeguards, metrics, limitations, and ownership. |

Remaining final-freeze checks are environment validations rather than missing
implementation: run full SparkyFitness TypeScript validation and `promtool` in
the deployment environment.
