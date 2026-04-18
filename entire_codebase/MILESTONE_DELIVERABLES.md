# SparkyFitness ML System — Milestone Deliverables Report

**Team size:** 3 members (Data, Training, Serving)  
**Deployment:** Chameleon Cloud KVM@TACC · VM IP `129.114.26.226`  
**Repository:** `entire_codebase/` (ML system) + `SparkyFitness/` (integrated app)  
**Date:** April 2026

---

## Joint Responsibilities (12/15)

### 1. Integrated System Deployed on Chameleon Cloud

The full ML system and SparkyFitness application run on a single Chameleon KVM@TACC VM, orchestrated via a **single multi-stage `Dockerfile`** and `docker-compose.yml`. The `pipeline` profile starts 15 services: one-shot setup/data/training jobs plus the steady-state app, serving, retraining, database, and monitoring services.

**Single Dockerfile — 4 named stages:**

```
Dockerfile
  base      ← shared: python:3.11-slim + OS packages + safeguarding/
  ├── data      ← data pipeline deps (pandas, soda, scipy, swiftclient)
  ├── training  ← XGBoost/MLflow/Ray deps
  ├── serving   ← FastAPI/Prometheus/asyncpg deps + /explain endpoint
  └── mlflow    ← mlflow==3.1.0 only (standalone, no shared base)
```

Each service in `docker-compose.yml` selects its stage via `target: data/training/serving/mlflow`. The `safeguarding/` module (fairness checker + SHAP explainability) lives in the shared `base` stage and is available to both training and serving images without duplication.

**Services deployed:**

| Container | Stage | Role | Port |
|-----------|-------|------|------|
| `sparkyfitness-setup` | — | Idempotently applies the ML route/UI patch to SparkyFitness | — |
| `sparkyfitness-db` | — | SparkyFitness application PostgreSQL | — |
| `sparkyfitness-server` | (SparkyFitness) | Node.js/Express API + recommendation bridge | 3010 |
| `sparkyfitness-frontend` | (SparkyFitness) | React/Vite UI with `RecipeRecommendations` panel | 3004 |
| `sparky-serving` | serving | FastAPI XGBoost ranking API + `/explain` endpoint | 8000 |
| `sparky-postgres` | — | ML PostgreSQL — user interactions + prediction logs + drift log | 5433 |
| `mlflow` | mlflow | MLflow 3.1.0 tracking + model registry | 5000 |
| `sparky-retrain-api` | training | HTTP webhook receiver for retraining pipeline | 8080 |
| `sparky-batch-pipeline` | data | One-shot training table compiler | — |
| `sparky-trainer` | training | One-shot XGBoost + fairness + registry pipeline | — |
| `sparky-drift-monitor` | data | KS-test drift loop + 90-day privacy cleanup | — |
| `prometheus` | — | Metrics scraping | 9090 |
| `grafana` | — | Dashboards | 3000 |
| `alertmanager` | — | Alert routing | 9093 |
| `postgres-exporter` | — | PostgreSQL metrics exporter | 9187 |

**Verified live endpoints on VM:**
- `http://129.114.26.226:3004` — SparkyFitness UI (recommendations visible)
- `http://129.114.26.226:8000/health` — ML serving health
- `http://129.114.26.226:8000/predict` — ranking inference
- `http://129.114.26.226:8000/explain` — SHAP per-prediction explanation
- `http://129.114.26.226:5000` — MLflow UI (fairness + SHAP artifacts visible)
- `http://129.114.26.226:9090` — Prometheus

---

### 2. End-to-End Automation Pipeline

The system is fully automated from data → training → serving with no manual SSH required for retraining, evaluation, or promotion.

**GitHub Actions CI/CD** (`.github/workflows/retrain.yml`) — 4-stage pipeline:

```
Stage 1: data-quality        → Soda checks (23 checks) on train/val/test CSVs
                               + drift report (no trigger in CI)
Stage 2: train-and-register  → 5-step pipeline:
                                 Step 1: Soda data quality gate
                                 Step 2: Patch config with data paths
                                 Step 3: XGBoost LambdaRank training
                                 Step 3.5: Fairness gate + SHAP explainability
                                 Step 4: Quality gates + MLflow registration
                                 Step 5: Optional auto-promote
Stage 3: auto-promote        → Promote Staging→Production + hot-reload serving
Stage 4: rollback-on-failure → Auto-rollback if smoke test fails
```

**Triggers:**
- `push` to `main` when training config or source code changes
- Cron: every Sunday 02:00 UTC (weekly forced retraining)
- Manual `workflow_dispatch` with `auto_promote` flag
- HTTP webhook from drift monitor → `retrain-api:8080/trigger` (drift-triggered)

**Runtime automation chain:**
1. Drift monitor runs KS-test every 300 seconds across 19 features
2. >30% features with p < 0.05 → POST to `retrain-api:8080/trigger`
3. `retrain_pipeline.py` runs all 5 steps including fairness gate
4. Serving hot-reloads new Production model every 60 seconds (zero downtime)
5. Previous Production version archived — instant rollback available

---

### 3. ML Feature Integrated into SparkyFitness (Open Source Service)

A personalized meal recommendation feature was added to `CodeWithCJ/SparkyFitness` across the full stack:

**Frontend** (`SparkyFitnessFrontend/`):
- `src/pages/Foods/RecipeRecommendations.tsx` — React component: loading skeletons, meal cards with name, kcal/protein/carbs/fat badges, personalised reason string, score pill, Add to Diary / Save / Dismiss actions
- `src/hooks/Foods/useRecommendations.ts` — TanStack Query hooks: `useRecommendations`, `useRecommendationFeedback`, `useInvalidateRecommendations`
- `src/api/recommendations.ts` — `fetchRecommendations()` + `postRecommendationFeedback()`

**Backend** (`SparkyFitnessServer/`):
- `routes/recommendationRoutes.ts` — `GET /api/recommendations` (Zod validation) + `POST /api/recommendations/feedback`
- `services/recommendationService.ts` — builds 47-feature vector per candidate meal, calls ML `/predict`, falls back to protein-proximity heuristic if ML is unreachable (10-second `AbortSignal.timeout`)
- `models/recommendationRepository.ts` — user goals, recently-logged meal exclusion, candidate pool, recommendation cache + interaction log tables
- `schemas/recommendationSchemas.ts` — Zod schemas for all request/response types
- `apply_integration.py` — idempotent setup script used by the `sparkyfitness-setup` container to copy files and patch `SparkyFitnessServer.ts` / `Foods.tsx`

**Full data flow:**
```
User opens Foods page
→ GET /api/recommendations (Express)
→ recommendationService: fetch user goals + recent meals from DB
→ build 47-feature vectors for candidate meals
→ convert SparkyFitness UUID/string IDs to stable numeric surrogate IDs for the Python ML API
→ POST http://sparky-serving:8000/predict
→ XGBoost LambdaRank scores + ranks candidates
→ map predicted numeric IDs back to SparkyFitness meal UUIDs
→ top-N results returned with reason strings + nutrition data
→ RecipeRecommendations renders meal cards
→ User clicks "Add to Diary"
→ POST /api/recommendations/feedback (action: "logged")
→ logInteraction() writes to recommendation_interactions table
→ app-level feedback stored in recommendation_interactions
→ ML retraining loop currently uses serving `/feedback` plus `user_feedback` / `user_interactions`
```

---

### 4. Safeguarding Plan — Implemented Within the System

All six safeguarding principles are implemented with concrete, automated mechanisms wired into the running pipeline — not standalone scripts.

#### Fairness (wired into training pipeline)

**File:** `safeguarding/fairness_checker.py` called from `src/training/retrain_pipeline.py` Step 3.5

After every training run, the scored test set is automatically loaded and `run_fairness_check()` is called:
- Computes NDCG@10 for 6 dietary groups (vegetarian, vegan, gluten-free, dairy-free, low-sodium, low-fat)
- Checks allergen safety: <1% of top-5 recommendations for restricted users contain their allergen
- Groups with <50 users are skipped (insufficient statistical power)
- **Gate:** `fairness_passed` is passed as Gate 3 into `evaluate_and_register()` — if it fails, the model is **not registered** to MLflow
- Results logged to MLflow as `fairness_results.json`; run tagged `fairness_passed=True/False`

#### Explainability (wired into training pipeline + serving API)

**Files:** `safeguarding/explainability.py` called from `retrain_pipeline.py` (global) and `app_production.py` (per-request)

- **Post-training (global):** `Explainer.global_feature_importance()` called in Step 3.5; SHAP mean |SHAP| per feature logged to MLflow as `explainability/global_feature_importance.json` + `shap_summary.png`
- **Per-request (live):** `POST /explain` endpoint in `app_production.py`; accepts any feature dict, returns top-10 SHAP contributions with human-readable text ("This recipe was recommended because: your protein target, cooking history matched")
- `Explainer` instance built on model load and rebuilt automatically on every hot-reload
- Rule-based fallback when SHAP unavailable (no latency impact on `/predict`)

#### Transparency (wired into serving API + MLflow)

- Every `/predict` response includes `model_version` and `model_source` fields
- Every training run logs: all hyperparameters, `dataset_metadata.json`, `run_summary.json`, `quality_gate_results.json`, `fairness_results.json`, `shap_summary.png`, config YAML, XGBoost model artifact
- MLflow UI at `:5000` shows complete lineage: training run → data version → model version → serving deployment

#### Privacy (wired into drift monitor)

**File:** `src/data/drift_monitor.py:cleanup_old_inference_features()` called every monitoring cycle

- User history sent to ML API as 6 PCA components — raw recipe IDs never leave SparkyFitness
- User and meal IDs passed to the ML API as stable numeric surrogate IDs; no email/name/device data is sent to the ML service
- **90-day retention enforced:** `cleanup_old_inference_features()` runs every 300 seconds, deletes `inference_features` rows older than 90 days, and logs the count deleted
- Inference features are now captured on every `/predict` call: `prediction_logger.log_features()` called alongside `log_batch()`, populating the `inference_features` table the drift monitor reads from

#### Accountability (wired into CI/CD + MLflow Registry)

- Every model in the Registry has `quality_gate_status`, `fairness_passed`, `registered_at`, `ndcg_at_10` tags
- Promotion Staging → Production logged in MLflow model description with timestamp and previous version number
- CI `production` environment requires approval before promotion
- `rollback_production()` archives current version and logs the replacement in the model description — no version is ever deleted

#### Robustness (wired into serving + alerting)

- 14 Prometheus alert rules are defined for serving, infrastructure, retraining, and data quality monitoring
- KS-test drift monitor: 19 features, every 300 seconds, >30% drift triggers retraining webhook
- Model fallback: if MLflow Registry unreachable, serves from local `MODEL_FALLBACK_PATH` file
- CI Stage 4 auto-rollback if smoke test fails after promotion

---

## Data Team Member (3/15)

### Deliverable D1: Data Quality at Ingestion

**Implementation:** `src/data/run_soda_checks.py` + `configs/data/soda_checks.yml`

Soda Core checks block the pipeline if any check fails. Integrated as Step 1 of `retrain_pipeline.py` — training never starts on bad data.

**Checks on `enriched_recipes`** (Food.com Kaggle, 230k+ recipes):

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
| Min review count | ≥ 3 |

**Checks on train/val/test splits:**

| Check | Train | Val | Test |
|-------|-------|-----|------|
| Min row count | > 100,000 | > 5 | > 5 |
| Null `user_id` | = 0 | — | — |
| Null `recipe_id` | = 0 | — | — |
| Null `label` | = 0 | = 0 | = 0 |
| Label balance | avg(label) 0.1–0.98 | — | — |

**Additional runtime checks** (`drift_monitor.py:check_ingestion_quality`):
- Column null rate > 50% → fails
- Raw file row count < 100 → fails
- Zero-variance column detection (pipeline bug signal)
- Temporal leakage: `max(train.date) < min(test.date)`

---

### Deliverable D2: Training Set Compilation Quality

**Implementation:** `src/data/build_training_table.py` + `drift_monitor.py:check_training_set_quality`

**Pipeline steps:**
1. Recipe enrichment: parse 7 macro nutrients, map 80+ cuisine tags → 20 categories (`ETHNICITY_MAP`), flag 9 allergens from ingredient text
2. Interaction filtering: rating ≥ 4 → `label=1`; drop users with <5 interactions
3. User feature derivation: cooking history → 6-dim PCA (`HISTORY_PCA_COMPONENTS=6`); user goal features from SparkyFitness profile
4. Time-stratified splits: train/val/test split by date, not random — prevents temporal leakage

**Manifest** (`output/manifest.json`): records split sizes, date ranges, feature count, SHA-256 of source files. Committed to repo; push triggers CI retraining.

**Training set quality checks** (runtime, in `check_training_set_quality`):
- Label positive rate must be 5%–95%
- Zero-variance numeric columns detected
- Negative calorie values detected
- File existence validated for all three splits

**Feature vector:** 47 features — recipe attributes, macros, allergen flags, user goal targets, dietary restriction flags, 6 history PCA components (full list in `app_production.py:FEATURE_COLUMNS`)

---

### Deliverable D3: Live Inference Drift Monitoring

**Implementation:** `src/data/drift_monitor.py`

**Mechanism:**
- Reference: 5,000-row sample from `train.csv` (loaded on startup)
- Live data: last 24 hours from `inference_features` PostgreSQL table — **now populated on every `/predict` call** via `prediction_logger.log_features()`
- Test: Kolmogorov-Smirnov two-sample test (`scipy.stats.ks_2samp`) on each of 19 numeric features
- Decision: drift if >30% features have p-value < 0.05 (env: `DRIFT_THRESHOLD=0.05`)
- Interval: every 300 seconds (env: `CHECK_INTERVAL_SECONDS=300`)
- Minimum: requires ≥100 live predictions before running KS-test

**19 monitored features:**
`minutes`, `n_ingredients`, `n_steps`, `avg_rating`, `n_reviews`, `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrate`, `daily_calorie_target`, `protein_target_g`, `carbs_target_g`, `fat_target_g`, `history_pc1`, `history_pc2`, `history_pc3`

**Per-feature result stored to `drift_log` table:**
`feature_name | ks_statistic | p_value | drift_detected | threshold | timestamp`

**Privacy retention:** `cleanup_old_inference_features()` runs every 300-second cycle, deleting `inference_features` rows older than 90 days.

**Action on drift:** POST to `http://retrain-api:8080/trigger` → full 5-step retraining pipeline including fairness gate.

---

## Training Team Member (3/15)

### Deliverable T1: Evaluation Metrics

**Implementation:** `src/training/metric.py`, `src/training/train.py`

**Primary metric: NDCG@10** (Normalized Discounted Cumulative Gain at rank 10)

```
DCG@10  = Σ (2^label_i - 1) / log2(rank_i + 2)   for rank i in 1..10
NDCG@10 = DCG@10 / IDCG@10   (IDCG = ideal ordering)
Final   = mean NDCG@10 across all users in test set
```

**Baseline comparison:** Every run evaluates a popularity-based baseline (`run_baseline_popularity`) — ranks recipes by total positive interactions. The XGBoost model must beat this baseline.

**Achieved metric:** NDCG@10 = **0.8148** on Food.com test split.

**All metrics logged to MLflow per run:**
`ndcg_at_10`, `wall_time_seconds`, dataset feature count (via `dataset_metadata.json`)

---

### Deliverable T2: Quality Gates (Only Passing Models Registered)

**Implementation:** `src/training/model_registry.py:evaluate_and_register()`

Three quality gates — all must pass for registration:

| Gate | Condition | Default Threshold |
|------|-----------|-------------------|
| 1. NDCG absolute | `ndcg_at_10 >= NDCG_THRESHOLD` | 0.55 (env: `NDCG_THRESHOLD`) |
| 2. Improvement over production | `ndcg >= prod_ndcg - IMPROVEMENT_THRESHOLD` | must not regress >0.01 |
| 3. Fairness | `fairness_passed=True` from `run_fairness_check()` | per-group NDCG ≥ 80% of overall + allergen safety <1% |

**Gate failure:** failed gates logged to MLflow as `quality_gate_results.json`; run tagged `quality_gate_status=FAILED`; model **not registered** — serving continues with existing Production version.

**Gate pass:** model registered to Registry as Staging; run tagged `quality_gate_status=PASSED` + `registry_version=N`; Staging → Production requires separate promotion step (CI Stage 3 or manual).

**Previous Production archived** (not deleted) on every promotion → instant rollback available.

---

### Deliverable T3: MLflow Experiment Tracking

**Implementation:** `src/training/train.py`, `src/training/mlflow_utils.py`, `src/training/model_registry.py`

**Every training run logs:**

| Artifact / Tag / Metric | Content |
|--------------------------|---------|
| `mlflow.log_metrics()` | `ndcg_at_10`, `wall_time_seconds` |
| `dataset_metadata.json` | feature count, all feature column names |
| `run_summary.json` | run name, candidate name, all metrics |
| `quality_gate_results.json` | all 3 gate pass/fail with values and thresholds |
| `fairness_results.json` | per-group NDCG@10, allergen safety violation rates |
| `explainability/global_feature_importance.json` | SHAP mean \|SHAP\| per feature |
| `explainability/shap_summary.png` | feature importance bar chart |
| `configs/training/xgb_ranker.yaml` | exact config used (reproducibility) |
| `mlflow.xgboost.log_model()` | XGBoost model in JSON format + input example |
| Tag: `quality_gate_status` | `PASSED` or `FAILED` |
| Tag: `fairness_passed` | `True` or `False` |
| Tag: `registry_version` | MLflow Registry version number |
| Tag: `ndcg_at_10` (registry) | metric at registration time |
| Tag: `registered_at` | ISO timestamp |

**MLflow tracking server:** `http://mlflow:5000` (Docker internal); `http://129.114.26.226:5000` externally  
**Artifact store:** shared Docker volume `mlflow-artifacts`  
**Model name:** `sparky-ranker` (env: `MLFLOW_MODEL_NAME`)  
**XGBoost objective:** `rank:ndcg` (LambdaRank)

---

## Serving Team Member (3/15)

### Deliverable S1: Model Output Monitoring

**Implementation:** `src/serving/app_production.py` (Prometheus metrics), `monitoring/alert_rules.yml`

**Custom Prometheus metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `sparky_prediction_score` | Histogram | Score distribution per request (buckets 0.0–1.0 in 0.1 steps) |
| `sparky_predictions_total` | Counter | Total predictions, labeled by `model_version` |
| `sparky_model_version` | Gauge | Currently loaded model version (numeric) |
| `sparky_model_reloads_total` | Counter | Number of hot-reloads |
| `sparky_model_loaded_timestamp_seconds` | Gauge | When the current model was loaded |
| `sparky_prediction_last_logged_timestamp_seconds` | Gauge | Last successful prediction-log write |
| `sparky_request_batch_size` | Histogram | Instances per `/predict` call (buckets 1,5,10,20,50,100,200) |

The retrain API also exposes:

| Metric | Type | Description |
|--------|------|-------------|
| `sparky_retrain_jobs_total` | Counter | Retrain jobs by status and reason |
| `sparky_retrain_failures_total` | Counter | Failed retrain jobs |
| `sparky_last_retrain_timestamp_seconds` | Gauge | Last successful retraining time |
| `sparky_retrain_job_running` | Gauge | Whether a retraining job is currently running |

Plus standard FastAPI metrics via `prometheus_fastapi_instrumentator`:
- `http_request_duration_seconds` (latency histogram by endpoint)
- `http_requests_total` (counter by status code)

**Score distribution alert** (`LowPredictionScores`): fires if p50 of `sparky_prediction_score` < 0.2 for 15 minutes — catches model degradation where the ranker stops producing confident scores.

---

### Deliverable S2: Operational Metrics and Alerting

**Implementation:** `monitoring/alert_rules.yml`, `monitoring/alertmanager.yml`

**14 alert rules:**

| Alert | Condition | Severity | For |
|-------|-----------|----------|-----|
| `ServingDown` | `up{job="sparky-serving"} == 0` | critical | 1 min |
| `HighPredictionLatency` | P95 /predict > 500ms | warning | 5 min |
| `HighErrorRate` | 5xx > 5% of requests | critical | 5 min |
| `LowPredictionScores` | Median score < 0.2 | warning | 15 min |
| `ZeroPredictionsServed` | No predictions in 10 min | warning | 10 min |
| `ModelVersionStale` | Loaded model age >24h | info | 1 hour |
| `PostgresDown` | DB unreachable | critical | 1 min |
| `MLflowDown` | MLflow unreachable | warning | 2 min |
| `HighMemoryUsage` | Container memory >85% | warning | 5 min |
| `HighDiskUsage` | Disk >85% (artifact store) | warning | 5 min |
| `PostgresHighConnections` | DB connections >80 | warning | 5 min |
| `RetrainingJobFailed` | Retrain failure in last hour | warning | immediate |
| `NoRetrainingIn7Days` | No retrain in 604,800 s | info | 1 hour |
| `PredictionLoggingLag` | No successful prediction-log write for >5 min | warning | 5 min |

---

### Deliverable S3: User Feedback Capture and Model Behavior Tracking

**Implementation:** `src/serving/app_production.py`, `src/serving/prediction_logger.py`, SparkyFitness integration

**Prediction logging (async, non-blocking):**  
Every `/predict` response is asynchronously logged to `prediction_log` table: `request_id`, `model_version`, `user_id`, `recipe_id`, `score`, `rank`, `timestamp`.

**Inference feature logging (for drift monitoring):**  
Every `/predict` call also writes raw feature values to `inference_features` table via `prediction_logger.log_features()` — this is what the drift monitor reads for KS-test distribution comparison.

**User feedback from SparkyFitness UI (4 signals):**
- `logged` — "Add to Diary" clicked
- `dismissed` — ✕ clicked
- `saved` — ★ clicked
- `viewed` — on impression

These reach `POST /api/recommendations/feedback` → `recommendationRepository.logInteraction()` → `recommendation_interactions` table.

**SHAP Explainability endpoint:**  
`POST /explain` returns top-10 SHAP feature contributions + human-readable reason string. `Explainer` instance built on model load; rebuilt on hot-reload. Graceful fallback to rule-based explanation if SHAP unavailable.

**Model version tracking in every response:**  
`model_version` + `model_source` in every `/predict` response enables per-version analysis on the serving side. App-side feedback is stored separately in `recommendation_interactions`; correlating it directly with `prediction_log` would require an explicit shared identifier to be added.

**Hot-reload without downtime:**  
`_poll_for_new_model()` background thread polls MLflow Registry every 60 seconds. On new Production version detection, swaps `_model` and `_explainer` globals under `threading.RLock`. In-flight requests complete with old model.

---

## Summary Table

| Deliverable | File(s) | Key Mechanism | Status |
|-------------|---------|---------------|--------|
| Ingestion quality | `run_soda_checks.py`, `soda_checks.yml` | 23 Soda checks, blocks pipeline on failure | ✓ Wired |
| Training set quality | `build_training_table.py`, `drift_monitor.py` | Label balance, temporal split, zero-variance detection | ✓ Wired |
| Live drift monitoring | `drift_monitor.py` | KS-test 19 features every 300s, >30% → retrain webhook | ✓ Wired |
| Privacy retention | `drift_monitor.py:cleanup_old_inference_features` | Deletes inference_features rows >90 days every cycle | ✓ Wired |
| Evaluation metric | `metric.py` | NDCG@10 = 0.8148 vs. popularity baseline | ✓ Wired |
| Fairness gate | `fairness_checker.py` → `retrain_pipeline.py` Step 3.5 → `model_registry.py` Gate 3 | Blocks registration if any dietary group >20% below overall NDCG | ✓ Wired |
| Allergen safety gate | `fairness_checker.py:check_allergen_safety` | <1% top-5 recs contain user allergen | ✓ Wired |
| MLflow tracking | `train.py`, `mlflow_utils.py` | Model + fairness + SHAP + config + gate results per run | ✓ Wired |
| SHAP global explainability | `explainability.py` → `retrain_pipeline.py` Step 3.5 | Logged to MLflow as artifact every training run | ✓ Wired |
| SHAP per-request | `explainability.py` → `app_production.py:/explain` | POST /explain returns feature contributions + text | ✓ Wired |
| Inference feature capture | `prediction_logger.log_features()` → `app_production.py` | Populates inference_features table for drift monitor | ✓ Wired |
| Output monitoring | `app_production.py` Prometheus | 5 custom metrics + score distribution alert | ✓ Wired |
| Operational alerts | `alert_rules.yml` | 14 alert rules critical/warning/info | ✓ Wired |
| Rollback | `model_registry.py:rollback_production` | Previous Production archived, instant restore | ✓ Wired |
| Safeguarding plan | `safeguarding/SAFEGUARDING_PLAN.md` | Covers all 6 principles with concrete mechanisms | ✓ Done |
| SparkyFitness integration | `sparkyfitness-integration/` | Full stack: React → Express → FastAPI → XGBoost | ✓ Done |
| End-to-end automation | `.github/workflows/retrain.yml` | 4-stage CI + weekly cron + drift-triggered retraining | ✓ Done |
| Single Dockerfile | `Dockerfile` | 4-stage multi-stage build, replaces 4 separate files | ✓ Done |
| Chameleon deployment | `CHAMELEON_DEPLOYMENT.md` | Step-by-step guide with all safeguarding verification steps | ✓ Done |
