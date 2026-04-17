# SparkyFitness ML System — Milestone Deliverables Report

**Team size:** 3 members (Data, Training, Serving)  
**Deployment:** Chameleon Cloud KVM@TACC · VM IP `129.114.26.226`  
**Repository:** `entire_codebase/` (ML system) + `SparkyFitness/` (integrated app)  
**Date:** April 2026

## Scope Clarification

This milestone submission is written for a **3-person team**:
- **Data**
- **Training**
- **Serving**

Accordingly, the integrated system is packaged with **Docker Compose** rather than Kubernetes.
The core repository is `entire_codebase/`, and the open-source service integration is provided in
`sparkyfitness-integration/` for the SparkyFitness application.

## Demo Checklist

These are the specific items to call out during the demo because they map directly to the milestone prompt.

### Bringing up the system

```bash
cd entire_codebase
docker compose --profile pipeline up -d
docker compose --profile data up -d data-generator
python scripts/smoke_test.py --url http://localhost:8000
```

This sequence brings up:
- data preparation and batch compilation
- model retraining and registration
- serving and monitoring
- the production-traffic replay script

### "Production data" script that hits service endpoints

`src/data/data_generator.py` is the production-emulation script. It replays held-out rows as live traffic by:
- building multi-candidate `POST /predict` requests against the serving API
- sending observed outcomes back through `POST /feedback`
- writing a local audit CSV in `output/generated_interactions.csv`

This demonstrates that production traffic flows through the same interfaces used by the deployed model service.

### Using the ML feature within the open-source service

The ML recommendation feature is implemented for SparkyFitness in:
- `sparkyfitness-integration/SparkyFitnessServer/`
- `sparkyfitness-integration/SparkyFitnessFrontend/`

In the regular user flow:
- the Foods page requests recommendations from the SparkyFitness backend
- the backend calls the ML serving API
- ranked recommendations are rendered in the app UI

### How feedback is captured

Two feedback paths are implemented:
- **ML-system path:** serving `POST /feedback` writes explicit outcomes to `user_feedback` and updates `user_interactions`
- **SparkyFitness path:** `recommendation_interactions` records app-level actions such as `logged`, `saved`, and `dismissed`

### How production data is saved for retraining

Production-serving activity is persisted in PostgreSQL:
- `prediction_log` stores ranked outputs
- `inference_features` stores live feature payloads for drift monitoring
- `user_feedback` and `user_interactions` store outcomes used by retraining

`src/data/batch_pipeline.py` reads the production interaction tables and merges them back into the next versioned training package.

### How retraining and redeployment work

When retraining is triggered:
1. `retrain_api.py` launches `retrain_pipeline.py`
2. Soda checks and training-quality checks run
3. training computes NDCG-based evaluation and safeguarding gates
4. passing models are registered in MLflow Staging
5. promotion moves the selected model to Production
6. `app_production.py` hot-reloads the new Production version automatically

### Safeguarding plan

The implemented safeguarding plan is in `safeguarding/SAFEGUARDING_PLAN.md` and is enforced by:
- fairness gates before registration
- SHAP-based explainability artifacts and `/explain`
- model-version transparency in serving responses and MLflow
- 90-day retention cleanup for `inference_features`
- monitoring, drift detection, promotion control, and rollback

### Role-owned evaluation and monitoring

- **Data:** ingestion quality checks, training-set quality checks, live drift monitoring
- **Training:** NDCG@10 evaluation, fairness gate, model-quality gates before registration
- **Serving:** operational metrics, output monitoring, feedback logging, hot-reload, rollback triggers

### Bonus-item integration

If a team member attempted a prior-stage bonus item, the demo should point to the integrated location in the unified system rather than presenting it as a standalone subsystem. In this repo, examples include:
- safeguarding logic wired into retraining and serving
- explainability exposed through `/explain`
- monitoring and rollback connected to the deployed serving stack

---

## Joint Responsibilities (12/15)

### 1. Integrated System Deployed on Chameleon Cloud

The full ML system and SparkyFitness application run as 12 Docker containers on a single Chameleon KVM@TACC VM, orchestrated by `docker-compose.yml`.

**Services deployed:**

| Container | Role | Port |
|-----------|------|------|
| `sparkyfitness-server` | Node.js/Express API (auth, food logging, recommendation bridge) | 3001 |
| `sparkyfitness-frontend` | React/Vite UI with `RecipeRecommendations` panel | 3004 |
| `sparky-serving` | FastAPI XGBoost ranking API (`app_production.py`) | 8000 |
| `sparky-db` | PostgreSQL — user data + prediction logs + drift log | 5433 |
| `mlflow` | MLflow 3.1.0 tracking + model registry | 5000 |
| `sparky-retrain-api` | HTTP webhook receiver triggering retraining pipeline | 8080 |
| `sparky-drift-monitor` | KS-test drift loop (5-min interval) | — |
| `prometheus` | Metrics scraping | 9090 |
| `grafana` | Dashboards | 3000 |
| `alertmanager` | Alert routing + rollback webhooks | 9093 |
| `mlflow-artifact-store` | Shared artifact volume mount | — |
| `soda-checks` (profile) | On-demand data quality runner | — |

**Key verified endpoints (live on VM):**
- `http://129.114.26.226:3004` — SparkyFitness web app (login, food diary, recommendations)
- `http://129.114.26.226:8000/health` — ML serving health
- `http://129.114.26.226:8000/predict` — ranking inference
- `http://129.114.26.226:5000` — MLflow UI
- `http://129.114.26.226:9090` — Prometheus
- `http://129.114.26.226:3000` — Grafana

**Access for others:** Share the VM IP. Anyone with network access to the Chameleon lease can reach all ports above directly. Chameleon leases run 7 days; training data persists in OpenStack Swift (`proj04-sparky-training-data` container) independent of lease expiry.

---

### 2. End-to-End Automation Pipeline

The system is fully automated from data → training → serving with no manual steps required.

**GitHub Actions CI/CD** (`.github/workflows/retrain.yml`) — 4-stage pipeline:

```
Stage 1: data-quality        → Soda checks on train/val/test CSVs
Stage 2: train-and-register  → XGBoost training + evaluate_and_register()
Stage 3: auto-promote        → Promote Staging→Production + hot-reload serving
Stage 4: rollback-on-failure → Auto-rollback if smoke test fails
```

**Triggers:**
- `push` to `main` when `output/manifest.json`, `configs/training/**`, or `src/training/**` change
- Cron: every Sunday 02:00 UTC (weekly forced retraining)
- Manual `workflow_dispatch` with optional `auto_promote` flag
- HTTP webhook from drift monitor (`src/data/drift_monitor.py` → `retrain-api:8080/trigger`)

**Automation chain (runtime):**
1. Drift monitor runs KS-test every 300 seconds
2. If >30% of 19 monitored features drift (p < 0.05), POST to `http://retrain-api:8080/trigger`
3. `retrain_api.py` launches `retrain_pipeline.py` in a background thread
4. Pipeline: download CSVs from Swift → Soda checks → train → quality gates → register to MLflow Staging
5. Serving polls MLflow Registry every 60 seconds; hot-reloads new Production model with zero downtime (`_poll_for_new_model()` in `app_production.py:137`)

**Data persistence:** Training CSVs uploaded to OpenStack Swift (`ingest_to_object_store.py`). CI downloads them via `python-swiftclient` before each training run.

---

### 3. ML Feature Integrated into SparkyFitness (Open Source Service)

A personalized meal recommendation feature was added to `CodeWithCJ/SparkyFitness` across the full stack:

**Frontend** (`SparkyFitnessFrontend/`):
- `src/pages/Foods/RecipeRecommendations.tsx` — React component with loading skeletons, meal cards (name, calories, protein/carbs/fat badges, recommendation reason, score pill), and Add to Diary / Save / Dismiss actions
- `src/hooks/Foods/useRecommendations.ts` — TanStack Query hooks: `useRecommendations`, `useRecommendationFeedback`, `useInvalidateRecommendations`
- `src/api/recommendations.ts` — API client: `fetchRecommendations(options)` + `postRecommendationFeedback(id, action)`
- Integrated into `Foods.tsx` page as a section above meal management

**Backend** (`SparkyFitnessServer/`):
- `routes/recommendationRoutes.ts` — Express router: `GET /api/recommendations` (with Zod query validation) + `POST /api/recommendations/feedback`
- `services/recommendationService.ts` — Calls ML serving API (`/predict` endpoint); builds 47-feature vector per candidate meal including user goals, allergen flags, history PCA components; falls back to protein-proximity heuristic if ML is unreachable
- `models/recommendationRepository.ts` — PostgreSQL queries: user goals, recently-logged meal IDs (exclusion window), candidate meal pool, recommendation cache (`recommendation_cache` table), interaction log (`recommendation_interactions` table)
- `schemas/recommendationSchemas.ts` — Zod schemas for all request/response types

**Data flow:**
```
User visits Foods page
→ GET /api/recommendations (Express)
→ recommendationService: fetch user goals + recent meals from DB
→ build 47-feature vectors for candidate meals
→ POST http://sparky-serving:8000/predict  (Docker internal network)
→ XGBoost scores + ranks candidates
→ top-N results returned with reason strings and nutrition data
→ RecipeRecommendations component renders meal cards
→ User clicks "Add to Diary" → POST /api/recommendations/feedback (action: "logged")
```

---

### 4. Safeguarding Plan

Safeguards operate at three stages: pre-training data validation, post-training fairness evaluation, and live inference monitoring.

**Fairness** (`safeguarding/fairness_checker.py`, wired into `src/training/retrain_pipeline.py` Step 3.5):
- After every training run, the scored test set is loaded and `run_fairness_check()` is called automatically
- Evaluates NDCG@10 for 6 dietary groups and allergen safety for 4 user-restriction/recipe-allergen pairs
- Criterion: per-group NDCG@10 ≥ 80% of overall; allergen violation rate < 1% in top-5 recommendations
- Results logged to MLflow as `fairness_results.json`; run tagged `fairness_passed=True/False`
- `fairness_passed` is passed as Gate 3 into `evaluate_and_register()` — a failing fairness check blocks model registration

**Explainability** (`safeguarding/explainability.py`, wired into training pipeline and serving API):
- **Post-training (global):** `Explainer.global_feature_importance()` called in Step 3.5 of `retrain_pipeline.py`; SHAP mean |SHAP| per feature logged to MLflow as `explainability/global_feature_importance.json` and `shap_summary.png`
- **Per-request (live):** `POST /explain` endpoint in `app_production.py` accepts any feature dict and returns top-10 SHAP contributions with human-readable explanation text ("This recipe was recommended because: your protein target, cooking history matched")
- `Explainer` instance created on model load/reload and stored as `_explainer` global; rebuilt automatically on hot-reload
- Graceful fallback to rule-based explanation if SHAP unavailable

**Privacy:**
- User history represented as 6 PCA components — raw recipe IDs not sent to ML API
- User ID passed as integer only; no PII in serving request body
- **90-day inference feature retention enforced:** `cleanup_old_inference_features()` called every monitoring cycle in `drift_monitor.py`; deletes rows from `inference_features` older than 90 days and logs the count

**Robustness:**
- Serving falls back to protein-proximity heuristic if ML API is unreachable (10-second `AbortSignal.timeout`)
- Model hot-reload without downtime; previous version archived (not deleted) for instant rollback
- **Inference features now captured on every `/predict` call:** `prediction_logger.log_features()` called alongside `log_batch()`, populating the `inference_features` table that the drift monitor reads from

---

## Data Team Member (3/15)

### Deliverable D1: Data Quality at Ingestion

**Implementation:** `src/data/run_soda_checks.py` + `configs/data/soda_checks.yml`

Soda Core checks run on every raw dataset before it enters the training pipeline. The checks validate both the enriched recipe catalogue and all three data splits.

**Checks on `enriched_recipes`** (source: Food.com Kaggle — 230k+ recipes):

| Check | Threshold |
|-------|-----------|
| Row count | > 50,000 |
| Null `recipe_id` | = 0 |
| Duplicate `recipe_id` | = 0 |
| Null `calories` | = 0 |
| Null `cuisine` | = 0 |
| Min calories | ≥ 0 |
| Max calories | < 500,000 |
| Avg calories | 200–800 kcal |
| Min prep time | > 0 minutes |
| Min ingredients | ≥ 1 |
| Min review count | ≥ 3 (ensures statistical reliability) |

**Checks on train/val/test splits:**

| Check | Train | Val | Test |
|-------|-------|-----|------|
| Min row count | > 100,000 | > 5 | > 5 |
| Null `user_id` | = 0 | — | — |
| Null `recipe_id` | = 0 | — | — |
| Null `label` | = 0 | = 0 | = 0 |
| Label balance | avg(label) 0.1–0.98 | — | — |

**Additional runtime checks** (`drift_monitor.py:check_ingestion_quality`):
- Null rate per column: fails if any column >50% null in raw CSVs
- Minimum row count enforcement on raw files
- Zero-variance column detection (potential pipeline bug signal)
- Temporal leakage guard: `max(train.date) < min(test.date)`

**Invocation:**
```bash
# In CI (blocks training if any check fails):
python src/data/run_soda_checks.py --train-csv output/train.csv --val-csv output/val.csv --test-csv output/test.csv

# On-demand via Docker:
docker compose --profile soda run --rm soda-checks
```

If any Soda check fails, the CI pipeline exits non-zero at Stage 1 — training never runs on bad data.

---

### Deliverable D2: Training Set Compilation Quality

**Implementation:** `src/data/build_training_table.py` + `src/data/drift_monitor.py:check_training_set_quality`

The training table is built from two Kaggle Food.com CSVs (RAW_recipes.csv, RAW_interactions.csv) through a deterministic pipeline:

**Pipeline steps:**
1. **Recipe enrichment** — parse nutrition list, derive macros (`total_fat_g`, `sugar_g`, `sodium_g`, `protein_g`, `saturated_fat_g`, `carbohydrate_g`), map cuisine from 80+ tag keywords (`ETHNICITY_MAP`), flag 9 allergens from ingredient text (`ALLERGEN_KEYWORDS`)
2. **Interaction filtering** — rating ≥ 4 → `label=1`, else `label=0`; drop users with <5 interactions (`MIN_USER_INTERACTIONS=5`)
3. **User feature derivation** — cooking history encoded as 6-dimensional PCA embedding (`HISTORY_PCA_COMPONENTS=6`); user goal features from SparkyFitness profile (daily calorie target, protein/carbs/fat targets, dietary flags)
4. **Time-stratified splits** — train/val/test split by date, not random; ensures val and test contain only future interactions (prevents temporal leakage)

**Output manifest:** `output/manifest.json` — records split sizes, date ranges, feature count, and SHA of source files. Committed to repo; changes trigger CI retraining.

**Training set quality checks** (`check_training_set_quality` in `drift_monitor.py:182`):
- Validates file existence for all three splits
- Checks label positive rate: fails if pos_rate < 5% or > 95% (severe imbalance)
- Detects zero-variance numeric columns (potential feature pipeline bug)
- Checks for negative calorie values

**Feature vector:** 47 features including recipe features, nutritional macros, allergen flags, user goal targets, dietary restriction flags, and 6 history PCA components (see full list in `app_production.py:FEATURE_COLUMNS`).

---

### Deliverable D3: Live Inference Drift Monitoring

**Implementation:** `src/data/drift_monitor.py`

A continuous monitoring loop compares the distribution of features arriving at the serving API against the training-time baseline distribution.

**Mechanism:**
- **Reference distribution:** 5,000-row sample from `train.csv` (loaded on startup from `TRAINING_BASELINE_PATH`)
- **Live distribution:** last 24 hours of inference features from `inference_features` PostgreSQL table (populated by the serving API on every prediction request)
- **Test:** Kolmogorov-Smirnov two-sample test (`scipy.stats.ks_2samp`) on each of 19 numeric features
- **Decision rule:** drift declared if >30% of features have p-value < 0.05 (`DRIFT_THRESHOLD=0.05`)
- **Check interval:** every 300 seconds (`CHECK_INTERVAL_SECONDS=300`)
- **Minimum samples:** requires ≥100 live predictions before testing (`MIN_SAMPLES_FOR_DRIFT=100`)

**19 monitored features:**
`minutes`, `n_ingredients`, `n_steps`, `avg_rating`, `n_reviews`, `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrate`, `daily_calorie_target`, `protein_target_g`, `carbs_target_g`, `fat_target_g`, `history_pc1`, `history_pc2`, `history_pc3`

**Per-feature result stored to `drift_log` table:**
```
feature_name | ks_statistic | p_value | drift_detected | threshold | timestamp
```

**Action on drift:** POST to `http://retrain-api:8080/trigger` with `auto_promote=false` (requires manual approval before production promotion). This closes the data→training feedback loop automatically.

**Invocation modes:**
```bash
python src/data/drift_monitor.py --loop       # continuous (Docker service)
python src/data/drift_monitor.py --once       # single check and exit
python src/data/drift_monitor.py --report-only  # CI: report without triggering
```

---

## Training Team Member (3/15)

### Deliverable T1: Evaluation Metrics

**Implementation:** `src/training/metric.py`, `src/training/train.py`, `src/training/model_registry.py`

**Primary metric: NDCG@10** (Normalized Discounted Cumulative Gain at rank 10)

The formula implemented in `metric.py:compute_ranking_metric`:

```
DCG@10  = Σ (2^label_i - 1) / log2(rank_i + 2)   for rank i in 1..10
NDCG@10 = DCG@10 / IDCG@10   (IDCG = ideal ordering)
Final   = mean NDCG@10 across all users in test set
```

NDCG@10 is the right metric here because:
- We show users a ranked list, not a binary prediction — rank position matters
- `@10` matches the UI (6 recommendations shown; buffer for dismissed items)
- Rewards getting the best meals to the top positions, not just binary correctness

**Baseline comparison:** Every training run also evaluates a popularity-based baseline (`run_baseline_popularity` in `train.py:24`) — ranks recipes by total positive interactions. The XGBoost model must beat this baseline on the test set.

**Achieved metric:** NDCG@10 = **0.8148** on the Food.com test split (reported in MLflow UI).

**All metrics logged to MLflow per run:**
- `ndcg_at_10` — primary ranking metric
- `wall_time_seconds` — training wall time
- `unique_recipes_ranked` (baseline only) — coverage
- `feature_count` — logged via `dataset_metadata.json` artifact

---

### Deliverable T2: Quality Gates (Only Passing Models Registered)

**Implementation:** `src/training/model_registry.py:evaluate_and_register`

Five quality gates must all pass before a model is registered to MLflow:

| Gate | Condition | Default Threshold |
|------|-----------|-------------------|
| 1. NDCG absolute threshold | `ndcg_at_10 >= NDCG_THRESHOLD` | 0.55 (env: `NDCG_THRESHOLD`) |
| 2. Improvement over production | `ndcg_at_10 > prod_ndcg - IMPROVEMENT_THRESHOLD` | must not regress by >0.01 |
| 3. Fairness — per-group NDCG | group NDCG ≥ 80% of overall | `MAX_DEGRADATION=0.20` |
| 4. Allergen safety | <1% top-5 recommendations contain user allergens | `violation_rate < 0.01` |
| 5. Soda data quality | All Soda checks pass before training runs | zero failed checks |

**Gate failure behavior:**
- Failed gates are written to MLflow run as `quality_gate_results.json` artifact
- Run is tagged `quality_gate_status=FAILED`
- Model is NOT registered to the Registry — serving continues with the existing Production version
- CI pipeline reports the failure in the job summary

**Gate pass behavior:**
- Model registered to Registry as version N in **Staging** stage
- Run tagged `quality_gate_status=PASSED` + `registry_version=N`
- Staging → Production promotion is a separate explicit step (manual or via CI auto-promote job)
- Previous Production version is **archived** (not deleted), enabling instant rollback

**Rollback path:**
```python
# model_registry.py:rollback_production()
# Finds most recent Archived version → transitions back to Production
# Previous Production archived (not lost)
python -m src.training.model_registry rollback
```

---

### Deliverable T3: MLflow Experiment Tracking

**Implementation:** `src/training/train.py`, `src/training/mlflow_utils.py`, `src/training/model_registry.py`

**Every training run logs:**

| Artifact / Tag / Metric | Content |
|--------------------------|---------|
| `mlflow.log_metrics(metrics)` | `ndcg_at_10`, `wall_time_seconds` |
| `dataset_metadata.json` | feature count, feature column names |
| `run_summary.json` | run name, candidate name, all metrics |
| `quality_gate_results.json` | gate pass/fail with values and thresholds |
| `configs/training/xgb_ranker.yaml` | exact config used (reproducibility) |
| `mlflow.xgboost.log_model(...)` | serialized XGBoost model in JSON format |
| `input_example` | 5 training rows (schema documentation) |
| Tag: `quality_gate_status` | `PASSED` or `FAILED` |
| Tag: `trainer_backend` | `ray_train` if Ray used |
| Tag: `registry_version` | MLflow Registry version number (if registered) |
| Tag: `ndcg_at_10` (on registry version) | metric at registration time |
| Tag: `registered_at` | ISO timestamp |

**MLflow experiment structure:**
- Experiment name from config (`experiment_name` field in `xgb_ranker.yaml`)
- Model registered under name `sparky-ranker` (env: `MLFLOW_MODEL_NAME`)
- Tracking server: `http://mlflow:5000` (within Docker network); accessible at `http://129.114.26.226:5000` externally
- Artifact store: shared Docker volume `mlflow-artifacts` (`--default-artifact-root /mlflow-artifacts`)

**Model Registry stages used:**
- `Staging` — auto-assigned after quality gates pass
- `Production` — assigned by CI auto-promote job or manual command
- `Archived` — previous Production version (rollback target)

**XGBoost hyperparameters (from config):**
- Objective: `rank:ndcg` (LambdaRank)
- `n_estimators`: configurable (default 200)
- `max_depth`: configurable (default 6)
- `learning_rate`: configurable (default 0.1)
- `eval_metric`: `ndcg@10`

---

## Serving Team Member (3/15)

### Deliverable S1: Model Output Monitoring

**Implementation:** `src/serving/app_production.py` (Prometheus metrics), `monitoring/alert_rules.yml`

**Custom Prometheus metrics exposed by serving API:**

| Metric | Type | Description |
|--------|------|-------------|
| `sparky_prediction_score` | Histogram | Distribution of prediction scores per request (buckets: 0.0–1.0 in 0.1 steps) |
| `sparky_predictions_total` | Counter | Total predictions served, labeled by `model_version` |
| `sparky_model_version` | Gauge | Currently loaded model version (numeric) |
| `sparky_model_reloads_total` | Counter | Number of live model hot-reloads |
| `sparky_request_batch_size` | Histogram | Instances per /predict call (buckets: 1,5,10,20,50,100,200) |

Plus standard FastAPI metrics via `prometheus_fastapi_instrumentator`:
- `http_request_duration_seconds` (latency histogram by endpoint)
- `http_requests_total` (counter by status code)

**Score distribution alert** (`alert_rules.yml:LowPredictionScores`):
- Fires if median prediction score (p50 of `sparky_prediction_score`) < 0.2 for 15 minutes
- This catches model degradation where the model stops confidently ranking any recipe

**Grafana dashboards** (`monitoring/grafana/`):
- P50/P95/P99 latency by endpoint
- Requests per second
- Error rate (4xx/5xx)
- Prediction score distribution over time
- Model version gauge (shows which version is live)
- Container memory/CPU per service

---

### Deliverable S2: Operational Metrics and Alerting

**Implementation:** `monitoring/alert_rules.yml`, `monitoring/alertmanager.yml`

**Alert rules (Prometheus → AlertManager):**

| Alert | Condition | Severity | For |
|-------|-----------|----------|-----|
| `ServingDown` | `up{job="sparky-serving"} == 0` | critical | 1 min |
| `HighPredictionLatency` | P95 /predict latency > 500ms | warning | 5 min |
| `HighErrorRate` | 5xx rate > 5% of all requests | critical | 5 min |
| `LowPredictionScores` | Median score < 0.2 | warning | 15 min |
| `ZeroPredictionsServed` | No predictions in 10 min | warning | 10 min |
| `ModelVersionStale` | Model unchanged for 24h | info | 1 hour |
| `PostgresDown` | DB unreachable | critical | 1 min |
| `MLflowDown` | MLflow unreachable | warning | 2 min |
| `HighMemoryUsage` | Container memory > 85% of limit | warning | 5 min |
| `HighDiskUsage` | Disk > 85% (MLflow artifacts) | warning | 5 min |
| `RetrainingJobFailed` | Retrain failure in last hour | warning | immediate |
| `NoRetrainingIn7Days` | No retrain in 604800 seconds | info | 1 hour |
| `PredictionLoggingLag` | Feedback loop lag > 5 min | warning | 5 min |

**Rollback trigger:** `RetrainingJobFailed` alert routes to AlertManager which posts to the retrain-api webhook. The CI pipeline's Stage 4 (`rollback-on-failure`) also triggers automatically if the smoke test fails after a promotion.

---

### Deliverable S3: User Feedback Capture and Model Behavior Tracking

**Implementation:** `src/serving/app_production.py`, `src/serving/prediction_logger.py`, SparkyFitness integration

**Prediction logging (async, non-blocking):**
Every `/predict` response is asynchronously logged to PostgreSQL `prediction_log` table via `PredictionLogger.log_batch()`. This captures: `request_id`, `model_version`, `user_id`, `recipe_id`, `score`, `rank`, `timestamp`.

**User feedback from SparkyFitness UI:**
The `RecipeRecommendations` component sends one of four feedback signals when users interact with cards:
- `logged` — user clicked "Add to Diary"
- `dismissed` — user clicked ✕
- `saved` — user clicked ★ (save for later)
- `viewed` — (can be sent on impression)

These reach `POST /api/recommendations/feedback` (Express) → `recommendationRepository.logInteraction()` → `recommendation_interactions` table in PostgreSQL.

**Serving API `/feedback` endpoint** (`app_production.py:303`):
Also accepts direct feedback from other clients, stored via `PredictionLogger.log_feedback()`.

**Model version tracking in responses:**
Every `/predict` response includes `model_version` and `model_source` fields. This allows per-version analysis: if a new model version causes a drop in `logged` actions or a spike in `dismissed` actions, it is detectable by joining `prediction_log` with `recommendation_interactions` on `request_id`.

**Hot-reload without downtime:**
`_poll_for_new_model()` runs in a background thread, checking MLflow Registry every 60 seconds. On detecting a new Production version, it calls `loader.load_production_model()` and swaps the global `_model` reference under a `threading.RLock`. In-flight requests complete with the old model; new requests use the new one.

**Admin endpoints:**
- `POST /admin/reload` — force immediate model reload
- `GET /admin/model-info` — current version, source, full feature list
- `GET /health` — `healthy` / `degraded` status with model version

---

## Summary Table

| Deliverable | File(s) | Key Threshold / Mechanism |
|-------------|---------|--------------------------|
| Ingestion quality | `src/data/run_soda_checks.py`, `configs/data/soda_checks.yml` | 11 checks on recipes, 4 on each split |
| Training set quality | `src/data/build_training_table.py`, `drift_monitor.py:check_training_set_quality` | Label balance, temporal split, zero-variance detection |
| Live drift monitoring | `src/data/drift_monitor.py` | KS-test, 19 features, p<0.05, >30% → retrain trigger |
| Evaluation metric | `src/training/metric.py` | NDCG@10 = 0.8148 vs. popularity baseline |
| Quality gates | `src/training/model_registry.py:evaluate_and_register` | 5 gates; failed → not registered |
| MLflow tracking | `src/training/train.py`, `mlflow_utils.py` | Model artifact + metrics + config + gate results per run |
| Output monitoring | `src/serving/app_production.py` | 5 custom Prometheus metrics + score distribution alert |
| Operational alerts | `monitoring/alert_rules.yml` | 13 alert rules, critical/warning/info |
| Feedback + rollback | `src/serving/prediction_logger.py`, `model_registry.py:rollback_production` | Async prediction log, 4 UI feedback actions, instant rollback |
| Safeguarding | `safeguarding/fairness_checker.py`, `safeguarding/explainability.py` | Fairness across 15 groups, SHAP explainability, privacy via PCA |
| Integration | `sparkyfitness-integration/` → `SparkyFitness/` | Full stack: React → Express → FastAPI → XGBoost |
| Automation | `.github/workflows/retrain.yml` | 4-stage CI + weekly cron + drift-triggered retraining |
