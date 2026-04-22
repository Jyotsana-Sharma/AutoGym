# Demo Video Guide

This guide is a practical recording script for the final system implementation
demo. Target length: 20-30 minutes. Use the Chameleon VM, not a local laptop,
for the live portions.

Before recording, avoid showing secrets. Keep `.env` closed or blur the terminal
if you display environment variables.

## Recommended Video Flow

### 0. Opening Context (1 minute)

Say:

> We are a 3-person team, so Platform/DevOps work is shared across the Data,
> Training, and Serving roles. The system is a personalized recipe
> recommendation feature integrated into SparkyFitness. It runs as a unified
> Docker Compose deployment on Chameleon.

Show:

```bash
cd ~/sparky-ml
git rev-parse --short HEAD
docker compose version
```

Briefly point to:

- `docker-compose.yml`
- `src/data/`
- `src/training/`
- `src/serving/`
- `sparkyfitness-integration/`
- `monitoring/`
- `safeguarding/`

### 1. Bring Up the System (3-4 minutes)

If starting from scratch:

```bash
cd ~/sparky-ml
docker compose --profile pipeline up -d --build
docker compose --profile pipeline ps
```

If the system is already bootstrapped:

```bash
cd ~/sparky-ml
docker compose --profile runtime up -d
docker ps --filter "name=sparky-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Explain that `pipeline` includes one-shot data/training jobs, while `runtime`
is the steady-state production stack.

Show health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8080/health
curl http://localhost:5000/health
curl -I http://localhost:3004
```

Expected interpretation:

- Serving is healthy and has a model version/source.
- Retrain API is healthy.
- MLflow responds `OK` at `/health`.
- SparkyFitness frontend is reachable.

### 2. Emulate Production Data (3 minutes)

Run the production traffic generator against the serving endpoint:

```bash
docker compose --profile data up -d data-generator
docker logs -f sparky-data-generator
```

Or run a bounded replay if you do not want an ongoing container:

```bash
docker compose --profile pipeline run --rm data-generator \
  python src/data/data_generator.py \
  --interactions-path /output/test.csv \
  --api-url http://sparky-serving:8000 \
  --rate 1.0 \
  --days 1 \
  --log-file /output/generated_interactions.csv
```

Explain:

- The script replays held-out interaction rows as production-like traffic.
- It sends multi-candidate `/predict` requests.
- It sends feedback to `/feedback`.
- It writes an audit CSV for the demo.
- Serving logs predictions and inference features into PostgreSQL.

Show that data is being captured:

```bash
docker exec -it sparky-postgres psql -U sparky -d sparky \
  -c "SELECT COUNT(*) AS predictions FROM prediction_log;"

docker exec -it sparky-postgres psql -U sparky -d sparky \
  -c "SELECT COUNT(*) AS feedback FROM user_feedback;"

docker exec -it sparky-postgres psql -U sparky -d sparky \
  -c "SELECT COUNT(*) AS inference_features FROM inference_features;"
```

### 3. Use the ML Feature in SparkyFitness (3-4 minutes)

Open:

```text
http://YOUR_VM_IP:3004
```

Show the regular SparkyFitness food/meal workflow where recipe
recommendations appear.

Explain:

- The frontend calls SparkyFitness backend recommendation routes.
- The backend builds candidate meal feature vectors.
- It calls the ML serving API at `ML_RECOMMENDATION_URL`.
- The ML API ranks candidates with the current Production model.
- The UI displays ranked recipes with scores and reason strings.

Then click at least one feedback action:

- Add to Diary / logged
- Save
- Dismiss

Show the feedback row:

```bash
docker exec -it sparkyfitness-db psql \
  -U "${SPARKY_FITNESS_DB_USER:-sparky}" \
  -d "${SPARKY_FITNESS_DB_NAME:-sparkyfitness_db}" \
  -c "SELECT action, COUNT(*) FROM recommendation_interactions GROUP BY action;"
```

Then show that ML-side feedback also exists:

```bash
docker exec -it sparky-postgres psql -U sparky -d sparky \
  -c "SELECT action, COUNT(*) FROM user_feedback GROUP BY action;"
```

### 4. Show Production Data Saved for Retraining (2-3 minutes)

Show the batch pipeline and manifest:

```bash
docker compose --profile pipeline run --rm batch-pipeline
docker compose --profile pipeline run --rm batch-pipeline \
  sh -c "ls -lh /output && cat /output/manifest.json"
```

Explain:

- The batch pipeline merges Food.com external data with production
  `user_interactions`.
- It creates versioned train/val/test splits.
- It uses temporal splitting to avoid leakage.
- The manifest records split sizes, hashes, timestamps, and label balance.

For the Data role, also show quality checks:

```bash
docker compose --profile pipeline run --rm batch-pipeline \
  python src/data/run_soda_checks.py \
  --data-dir /output \
  --checks configs/data/soda_checks.yml
```

### 5. Retraining and Redeployment (4-5 minutes)

Show current production model:

```bash
curl -s http://localhost:8080/model/production | python3 -m json.tool
```

Trigger retraining:

```bash
curl -s -X POST http://localhost:8080/trigger \
  -H "Content-Type: application/json" \
  -d '{"reason":"demo_manual","config":"configs/training/xgb_ranker.yaml","auto_promote":false}' \
  | python3 -m json.tool
```

Poll status:

```bash
watch -n 10 'curl -s http://localhost:8080/status | python3 -m json.tool'
```

When it finishes, show:

- result status
- MLflow run id
- model quality metrics
- whether model registered
- gate results

Open MLflow:

```text
http://YOUR_VM_IP:5000
```

Show:

- experiment runs
- metrics such as `ndcg_at_10`
- artifacts such as `quality_gate_results.json`
- fairness results
- SHAP/explainability artifacts if present
- model registry version/stage

Promote or explain why promotion is intentionally manual:

```bash
curl -s -X POST http://localhost:8080/promote \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool
```

Then force or wait for serving reload:

```bash
curl -s -X POST http://localhost:8000/admin/reload | python3 -m json.tool
curl -s http://localhost:8000/admin/model-info | python3 -m json.tool
```

Explain that serving hot-reloads Production model versions from MLflow Registry
and keeps rollback available through archived versions.

### 6. Safeguarding Plan (3 minutes)

Open:

- `safeguarding/SAFEGUARDING_PLAN.md`
- MLflow artifacts
- `/explain` endpoint
- Prometheus/Grafana alerts

Show one live explainability call:

```bash
python3 scripts/smoke_test.py --url http://localhost:8000
```

Or:

```bash
curl -s -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"instance":{"user_id":1,"recipe_id":33096,"calories":420,"protein":25,"minutes":30,"avg_rating":4.2},"top_k":5}' \
  | python3 -m json.tool
```

Explain the six principles:

- Fairness: dietary-group NDCG and allergen safety gate before registration.
- Explainability: SHAP/global feature importance plus `/explain`.
- Transparency: model version/source in responses and MLflow lineage.
- Privacy: pseudonymized IDs and 90-day retention for inference features.
- Accountability: model registry stages, tags, run artifacts, and rollback trail.
- Robustness: drift monitoring, alerting, fallback model, rollback endpoint.

### 7. Role-Owned Evaluation and Monitoring (6-8 minutes total)

Each team member should speak for about 2 minutes.

#### Data Role

Show:

```bash
docker compose --profile pipeline run --rm batch-pipeline \
  python src/data/drift_monitor.py --once

docker exec -it sparky-postgres psql -U sparky -d sparky \
  -c "SELECT feature_name, ks_statistic, p_value, drift_detected, run_at FROM drift_log ORDER BY run_at DESC LIMIT 10;"
```

Explain:

- ingestion quality checks
- train/val/test compilation quality
- temporal leakage avoidance
- live inference drift
- retraining trigger on drift
- 90-day feature retention

If showing the Data bonus, show the integrated framework or object-storage piece
you claimed, such as Soda checks and Swift/versioned dataset manifests.

#### Training Role

Show:

```bash
curl -s http://localhost:8080/status | python3 -m json.tool
```

Open MLflow and show:

- multiple runs
- baseline vs. XGBoost candidate
- `ndcg_at_10`
- training cost metric such as wall time
- quality gate artifact
- fairness artifact
- registered model stage

Explain:

- model quality gate
- production-regression gate
- fairness gate blocks registration if failed
- retraining can be scheduled, drift-triggered, or manual

If showing the Training bonus, show the Ray training config/run if it was used:

- `configs/training/xgb_ranker_ray.yaml`
- Ray-related MLflow run or training output

#### Serving Role

Open:

```text
http://YOUR_VM_IP:9090/alerts
http://YOUR_VM_IP:3000
```

Grafana login:

```text
username: admin
password: value of GRAFANA_PASSWORD, default sparky_admin
```

Show:

- request latency
- prediction counts
- score distribution
- currently loaded model version
- retrain API metrics
- active/inactive alert rules

Run:

```bash
python3 scripts/benchmark.py --url http://localhost:8000
```

or:

```bash
python3 scripts/smoke_test.py --url http://localhost:8000
```

Explain:

- `HighPredictionLatency`
- `HighErrorRate`
- `LowPredictionScores`
- `PredictionLoggingLag`
- rollback webhook for critical alerts
- hot-reload and fallback model behavior

If showing the Serving bonus, show any non-lab serving or alerting integration
you claimed. In this repo, Grafana alert provisioning to `/alerts/rollback` is
the most concrete operational integration to show.

## Prometheus and Grafana Notes for the Demo

Prometheus is used to scrape and evaluate operational metrics. Grafana is used
to visualize the same Prometheus-backed metrics and provision critical alert
notifications.

Important known note: the current `monitoring/prometheus.yml` includes an
`mlflow` scrape target at `/metrics`. This MLflow server is healthy at `/health`
but does not expose Prometheus metrics at `/metrics`, so Prometheus may show
`mlflow (0/1 up)` and fire `MLflowDown`. Interpret this as a monitoring target
configuration limitation, not an MLflow outage. In the video, verify MLflow
health with:

```bash
curl http://localhost:5000/health
docker ps | grep mlflow
```

Then explain that model registry/tracking is healthy even if the Prometheus
`/metrics` scrape target is down.

Other alerts:

- `PredictionLoggingLag`: usually means no recent predictions or logging failed.
- `RetrainingJobFailed`: show `/status` to explain whether the last retraining
  failed due to data, MLflow, quality gates, or infrastructure.
- `NoRetrainingIn7Days`: expected until a successful retrain updates the metric.
- Serving alerts inactive is good: serving is up, low error, acceptable latency.

## Suggested 25-Minute Timeline

| Time | Segment |
|---|---|
| 0:00-1:00 | Project context and team roles |
| 1:00-4:30 | Bring up system and health checks |
| 4:30-7:30 | Production data generator and DB proof |
| 7:30-11:00 | SparkyFitness UI feature + feedback capture |
| 11:00-14:00 | Batch data for retraining + data quality |
| 14:00-19:00 | Retraining, MLflow, promotion/reload |
| 19:00-22:00 | Safeguarding plan |
| 22:00-28:00 | Data, Training, Serving role monitoring/evaluation |
| 28:00-30:00 | Wrap-up and known limitations |

## Final Checklist Before Recording

Run:

```bash
docker ps --filter "name=sparky-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
curl http://localhost:8000/health
curl http://localhost:8080/health
curl http://localhost:5000/health
python3 scripts/smoke_test.py --url http://localhost:8000
```

Open:

- SparkyFitness UI: `http://YOUR_VM_IP:3004`
- MLflow: `http://YOUR_VM_IP:5000`
- Prometheus: `http://YOUR_VM_IP:9090`
- Grafana: `http://YOUR_VM_IP:3000`

Record one browser window and one terminal. Keep terminal font large enough for
graders to read.
