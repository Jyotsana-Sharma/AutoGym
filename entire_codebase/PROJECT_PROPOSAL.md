# SparkyFitness ML System — Complete Pipeline Documentation

**Status: PRODUCTION DEPLOYED on Chameleon Cloud KVM@TACC**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Pipeline Stage Details](#4-pipeline-stage-details)
   - 4.1 [Raw Data Ingestion](#41-raw-data-ingestion)
   - 4.2 [Data Quality Gate (Soda Core)](#42-data-quality-gate-soda-core)
   - 4.3 [Synthetic Data Generation](#43-synthetic-data-generation)
   - 4.4 [Batch Dataset Compilation](#44-batch-dataset-compilation)
   - 4.5 [Model Training — XGBoost LambdaRank](#45-model-training--xgboost-lambdarank)
   - 4.6 [Safeguarding Gate (Fairness + Explainability)](#46-safeguarding-gate-fairness--explainability)
   - 4.7 [Quality Gates and MLflow Registration](#47-quality-gates-and-mlflow-registration)
   - 4.8 [Production Serving](#48-production-serving)
   - 4.9 [Inference Feature Logging](#49-inference-feature-logging)
   - 4.10 [Drift Monitoring and Retraining Loop](#410-drift-monitoring-and-retraining-loop)
   - 4.11 [CI/CD Pipeline (GitHub Actions)](#411-cicd-pipeline-github-actions)
5. [Safeguarding Plan](#5-safeguarding-plan)
   - 5.1 [Fairness](#51-fairness)
   - 5.2 [Explainability](#52-explainability)
   - 5.3 [Transparency](#53-transparency)
   - 5.4 [Privacy](#54-privacy)
   - 5.5 [Accountability](#55-accountability)
   - 5.6 [Robustness](#56-robustness)
6. [SparkyFitness Integration](#6-sparkyFitness-integration)
7. [Feature Schema (47 Dimensions)](#7-feature-schema-47-dimensions)
8. [API Contracts](#8-api-contracts)
9. [Infrastructure and Container Architecture](#9-infrastructure-and-container-architecture)
10. [Monitoring Stack](#10-monitoring-stack)
11. [Technology Stack](#11-technology-stack)
12. [Evaluation and Metrics](#12-evaluation-and-metrics)
13. [Data Strategy](#13-data-strategy)
14. [Deliverables](#14-deliverables)

---

## 1. Executive Summary

**SparkyFitness** is a production-grade, fully automated personalized recipe recommendation system built on a continuous learning flywheel. It ranks meal suggestions according to each user's nutritional goals, dietary restrictions, allergen constraints, and personal cooking history.

The system is deployed on **Chameleon Cloud (KVM@TACC)** as a suite of 12 Docker containers orchestrated by a single `docker-compose.yml`. It integrates with the open-source **SparkyFitness** nutrition tracking application, which calls the ML serving layer at inference time for every recipe recommendation in the regular user flow.

### Key Achievements

| Metric | Value |
|---|---|
| Model quality | NDCG@10 = **0.8148** on held-out test set |
| Feature vector | **47 dimensions** (recipe + nutritional + allergen + user history) |
| Inference latency | P50 < 50 ms, P99 < 100 ms |
| Training automation | 4-stage CI/CD with automatic model promotion |
| Retraining trigger | Automated via KS-test drift monitor or scheduled weekly |
| Safeguarding | Fairness gate + SHAP explainability + 90-day privacy retention — all wired into pipeline |
| Containers | 12 services, single multi-stage Dockerfile |
| Cloud storage | Chameleon OpenStack Swift — versioned training datasets |

---

## 2. Problem Statement

Generic recipe recommendation systems fail in two fundamental ways:

1. **Context Blindness**: They ignore individual nutritional targets, allergen sensitivities, and cooking time constraints.
2. **Static Personalization**: Collaborative filtering degrades over time without continuous retraining on evolving user behavior.

**SparkyFitness** addresses this by framing recipe recommendation as a **supervised learning-to-rank problem**. Each candidate recipe is scored against the full context of the user's profile, producing a personalized ranking that is nutritionally aware, allergen-safe, and preference-aligned — and automatically updated as user behavior evolves.

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL DATA                                                                  │
│  Kaggle Food.com: RAW_recipes.csv (230K) + RAW_interactions.csv (1.1M)         │
└───────────────────────────┬─────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  DATA PIPELINE  [containers: data-generator, batch-pipeline, drift-monitor]     │
│                                                                                 │
│  ingest_to_object_store.py                                                      │
│    → Schema validation + SHA-256 versioning                                     │
│    → Upload to Chameleon Swift: proj04-sparky-raw-data                          │
│                                                                                 │
│  build_training_table.py / batch_pipeline.py                                   │
│    → Cuisine mapping (100+ tags → 20 categories)                                │
│    → Allergen detection (9 allergen types)                                      │
│    → Nutrition parsing + label generation (rating ≥ 4 → positive)              │
│    → PCA cooking history (3 components)                                          │
│    → Time-stratified split (train/val/test — no leakage)                        │
│    → manifest.json (SHA-256 + git commit + timestamp)                           │
│    → Upload to Swift: proj04-sparky-training-data                               │
│                                                                                 │
│  data_generator.py → PostgreSQL (user_interactions)                             │
│    → Synthetic user profiles + realistic interaction replay                     │
│                                                                                 │
│  drift_monitor.py (every 300 seconds)                                           │
│    → KS-test on 19 numeric features (live vs. training baseline)                │
│    → Drift detected (>30% features) → POST /trigger to retrain-api              │
│    → 90-day inference feature retention (privacy)                               │
└───────────────────────────┬─────────────────────────────────────────────────────┘
                            │
                   train.csv / val.csv / test.csv
                        manifest.json
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE  [containers: trainer, retrain-api]                          │
│                                                                                 │
│  retrain_pipeline.py                                                            │
│    Step 1: Load versioned data from Swift (SHA-256 verified)                    │
│    Step 2: XGBoost LambdaRank training (rank:ndcg, NDCG@10)                    │
│    Step 3: Log all params + artifacts to MLflow                                 │
│    Step 3.5: SAFEGUARDING GATE                                                  │
│      → fairness_checker.py: calorie distribution parity by dietary group        │
│        (Cohen's d ≤ 0.5; fail → model NOT registered)                          │
│      → explainability.py: SHAP TreeExplainer global importance                 │
│        → logged as artifact to MLflow run                                       │
│    Step 4: Quality gates (NDCG threshold + fairness gate)                       │
│    Step 5: model_registry.py → MLflow Registry (Staging)                        │
│            → Staging → Production (auto-promote if all gates pass)              │
│                                                                                 │
│  retrain_api (port 8080)                                                        │
│    → POST /trigger → launches new training run                                  │
│    → Called by: drift_monitor.py, GitHub Actions, manual                        │
└───────────────────────────┬─────────────────────────────────────────────────────┘
                            │
                      Model in MLflow Registry
                      (Production stage)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SERVING LAYER  [container: serving, port 8000]                                 │
│                                                                                 │
│  app_production.py (FastAPI + Uvicorn, 2 workers)                               │
│    → GET  /health     → liveness + model version                                │
│    → POST /predict    → ranked recipe list                                      │
│         • assemble_features() → 47-dim float32 matrix                          │
│         • XGBoost DMatrix → booster.predict() → sort by score                  │
│         • prediction_logger.log_batch() → PostgreSQL (predictions)              │
│         • prediction_logger.log_features() → PostgreSQL (inference_features)   │
│    → POST /explain    → SHAP per-feature attribution for one instance           │
│         • Explainer.explain_prediction() → top-K feature importances           │
│    → GET  /metrics    → Prometheus scrape endpoint                              │
│                                                                                 │
│  Model hot-reload: checks MLflow Registry every 60 seconds                      │
│    → New Production version found → reload under threading.RLock               │
│    → Rebuild _explainer (SHAP TreeExplainer) after model reload                 │
│                                                                                 │
│  SparkyFitness calls POST /predict for every recipe recommendation              │
│    in the regular user flow via ML_RECOMMENDATION_URL env var                   │
└───────────────────────────┬─────────────────────────────────────────────────────┘
                            │
             inference_features logged per request
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MONITORING  [containers: prometheus, grafana, alertmanager, postgres-exporter] │
│                                                                                 │
│  Prometheus (port 9090) scrapes:                                                │
│    → sparky-serving /metrics (request count, latency, errors)                  │
│    → retrain-api /metrics                                                       │
│    → postgres-exporter (DB performance)                                         │
│                                                                                 │
│  Grafana (port 3000) dashboards:                                                │
│    → Inference latency P50/P95/P99                                              │
│    → Drift monitor results (from PostgreSQL drift_log table)                    │
│    → Training pipeline success/failure history                                  │
│                                                                                 │
│  Alertmanager (port 9093):                                                      │
│    → Alert on high error rate, latency > threshold, model staleness            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Pipeline Stage Details

### 4.1 Raw Data Ingestion

**File**: `src/data/ingest_to_object_store.py`
**Container**: `data` stage (no dedicated container — run as one-shot job)
**Output**: Versioned raw CSVs in Chameleon Swift `proj04-sparky-raw-data`

The ingestion script is the system's first quality checkpoint:

1. **Schema validation**: Enforces required columns for both `RAW_recipes.csv` and `RAW_interactions.csv`. Fails fast with clear error if schema changes (e.g., upstream Kaggle format change).
2. **SHA-256 hashing**: Computes content hashes of both files before upload. Stored as Swift object metadata and in `manifest.json` for full lineage.
3. **Synthetic expansion**: If dataset is below 5 GB, applies Gaussian noise (σ=0.05) to numeric columns and doubles the dataset (`EXPAND_FACTOR=2`) to simulate production-scale data volumes.
4. **Object storage upload**: Target container `proj04-sparky-raw-data` on Chameleon Cloud (CHI@TACC region). Container is set to public-read.

---

### 4.2 Data Quality Gate (Soda Core)

**File**: `src/data/run_soda_checks.py`
**Purpose**: Automated validation before any model training begins

Soda Core checks enforce data integrity as a hard gate:

**Checks on `enriched_recipes`**:
- `row_count > 50000` — minimum dataset size
- `missing_count(recipe_id) = 0` — no null primary keys
- `duplicate_count(recipe_id) = 0` — referential integrity
- `min(calories) >= 0` and `max(calories) < 500000` — nutritional sanity
- `avg(calories) between 200 and 800` — distribution sanity
- `min(n_reviews) >= 3` — minimum recipe credibility

**Checks on training splits**:
- Row count thresholds per split (train/val/test)
- Zero nulls in `user_id`, `recipe_id`, `label`
- Label balance: `avg(label) between 0.1 and 0.98` — prevents severely imbalanced training

If any check fails, the pipeline exits non-zero and GitHub Actions marks the step as failed — no model is trained on bad data.

Additionally, `drift_monitor.py` runs `check_training_set_quality()` at compile time to verify:
- No all-zero (zero-variance) feature columns (pipeline bug detection)
- No negative calorie values
- Label balance within bounds for each split

---

### 4.3 Synthetic Data Generation

**File**: `src/data/data_generator.py`
**Container**: `sparky-data-generator` (profile: `data`, `full`)
**Output**: PostgreSQL `user_interactions` table

The data generator simulates a production user base to enable offline retraining cycles without live traffic:

- Generates realistic user profiles with random allergens, dietary restrictions, and macro targets
- Assigns meal contexts based on hour of day (breakfast 5–10am, lunch 10–2pm, dinner 6–10pm, snack otherwise)
- Maps ratings to realistic actions: `rating ≥ 4 → cook`, `3–4 → cook/view/view`, `2–3 → view/skip/skip`, `< 2 → skip`
- Replays held-out Kaggle test interactions in chronological order at a configurable rate (default: 50 interactions/minute via `--rate 1.0 --days 7`)
- Writes all events to PostgreSQL `user_interactions` with timestamps

This table is the live interaction stream that feeds both:
- The batch pipeline (for feature engineering and label generation)
- The serving layer (for user feature cache population)

---

### 4.4 Batch Dataset Compilation

**File**: `src/data/batch_pipeline.py`
**Container**: `sparky-batch-pipeline` (profile: `data`, `retrain`, `full`)
**Output**: `train.csv`, `val.csv`, `test.csv`, `manifest.json` → Swift + shared volume

The batch pipeline merges historical data with new production interactions to build versioned training packages:

1. Load historical train/val/test CSVs from Swift (verified by SHA-256)
2. Query PostgreSQL `user_interactions` for new interactions since last run
3. Merge and apply full feature engineering:
   - Cuisine mapping (100+ Kaggle tags → 20 canonical categories)
   - Allergen detection (9 allergen types via keyword scanning)
   - Nutritional parsing (`calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrate`)
   - Aggregation: `avg_rating`, `n_reviews` from interaction join
   - Label generation: `rating ≥ 4 → label=1`
   - PCA cooking history: 6-component → **3 components** used in 47-feature schema (`history_pc1`, `history_pc2`, `history_pc3`)
4. Time-stratified split:
   - **Train**: All interactions before the last 14 days
   - **Test**: Interactions from day −14 to day −7
   - **Val**: Interactions from day −7 to present
5. Quality checks: temporal ordering (`train_max_date < test_min_date`), label balance, no nulls in critical columns
6. Generate `manifest.json` with SHA-256 hashes + Git commit SHA + timestamp
7. Upload to Swift `proj04-sparky-training-data`

**Manifest Schema**:
```json
{
  "pipeline_version": "1.0.0",
  "git_commit": "<40-char SHA>",
  "timestamp": "<ISO-8601>",
  "splits": {
    "train": { "path": "train.csv", "sha256": "...", "rows": 100000 },
    "val":   { "path": "val.csv",   "sha256": "...", "rows": 10000  },
    "test":  { "path": "test.csv",  "sha256": "...", "rows": 10000  }
  }
}
```

---

### 4.5 Model Training — XGBoost LambdaRank

**File**: `src/training/retrain_pipeline.py`
**Container**: `sparky-trainer` (profile: `retrain`, `full`)
**Config**: `configs/training/xgb_ranker.yaml`

#### Algorithm

**XGBoost LambdaRank** (`rank:ndcg` objective):

LambdaRank is a pairwise learning-to-rank algorithm that directly optimizes NDCG. It computes gradient updates using pairwise preference differences weighted by the change in NDCG from swapping each pair. This is superior to pointwise regression for ranking tasks because:
- The loss directly reflects the ranking quality metric (NDCG), not proxy metrics
- Group-based training naturally captures per-user ranking context
- XGBoost handles sparse/mixed feature types efficiently

#### Hyperparameters

```yaml
objective:        rank:ndcg
eval_metric:      ndcg@10
learning_rate:    0.05
n_estimators:     200
max_depth:        6
subsample:        0.9
colsample_bytree: 0.9
reg_lambda:       1.0
```

#### Training Steps (retrain_pipeline.py)

```
Step 1: Load versioned CSV data (SHA-256 verified against manifest)
        → PreparedFrames: feature matrix (N×47), group arrays, feature names

Step 2: XGBoost LambdaRank training
        → XGBRanker.fit(X_train, y_train, group=groups_train, eval_set=...)
        → Early stopping on NDCG@10 (patience=10 rounds)
        → Final NDCG@10 = 0.8148

Step 3: Log to MLflow
        → All YAML params (flattened: training.learning_rate, etc.)
        → NDCG@10 metric
        → Model artifact (JSON + booster)
        → Config file as artifact

Step 3.5: SAFEGUARDING GATE [see Section 4.6]

Step 4: Quality gates (model_registry.py)
        → Gate 1: NDCG@10 ≥ 0.55 (NDCG_THRESHOLD)
        → Gate 2: Test loss not worse than val loss by > 10%
        → Gate 3: fairness_passed (from Step 3.5)
        → ALL gates must pass → model registered to MLflow Registry

Step 5: Promote to Staging, then auto-promote to Production
```

**NDCG@10 = 0.8148** — substantially above the 0.55 threshold gate.

---

### 4.6 Safeguarding Gate (Fairness + Explainability)

**Files**: `safeguarding/fairness_checker.py`, `safeguarding/explainability.py`
**Called from**: `src/training/retrain_pipeline.py` Step 3.5
**Impact**: Blocking — failing fairness prevents MLflow registration

This is Step 3.5 inserted between training and registration. It is a hard blocker — not a warning.

#### Fairness Check

```python
# In retrain_pipeline.py Step 3.5:
test_df = pd.read_csv(test_csv)
booster = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
feature_cols = [c for c in test_df.columns if c not in ("user_id", "recipe_id", "label", ...)]
dmatrix = xgb.DMatrix(test_df[feature_cols].fillna(0).values.astype("float32"), feature_names=feature_cols)
test_df["score"] = booster.predict(dmatrix)
fairness_result = run_fairness_check(test_df)
fairness_passed = fairness_result["overall_passed"]
```

`fairness_checker.py` evaluates:
- Score distribution across dietary groups (`user_vegetarian`, `user_vegan`, `user_gluten_free`, `user_dairy_free`, `user_low_sodium`, `user_low_fat`)
- Uses **Cohen's d** — effect size between each dietary group and the general population
- **Threshold**: Cohen's d ≤ 0.5 (medium effect size) for each group
- Result logged to MLflow as `fairness_report.json` artifact

If `fairness_passed = False`, the model is **not registered** to the MLflow Registry. The training job exits with a non-zero code, GitHub Actions marks the step failed, and no promotion occurs.

#### SHAP Explainability

```python
# Also in Step 3.5, after fairness check:
explainer = Explainer(booster, feature_columns=feature_cols)
global_importance = explainer.get_global_importance(test_df[feature_cols])
mlflow.log_dict(global_importance, "shap_global_importance.json")
```

`explainability.py` uses `shap.TreeExplainer` to compute:
- **Global feature importance**: mean |SHAP value| across test set
- **Per-instance attribution**: available at runtime via `/explain` endpoint
- Results stored as artifact in every MLflow training run

---

### 4.7 Quality Gates and MLflow Registration

**File**: `src/training/model_registry.py`
**Called from**: `retrain_pipeline.py` Step 4–5

Three gates must all pass before a model is registered:

| Gate | Check | Threshold |
|---|---|---|
| Gate 1: Performance | `NDCG@10 ≥ NDCG_THRESHOLD` | 0.55 (configurable) |
| Gate 2: Overfit | `test_loss ≤ val_loss × 1.1` | Max 10% degradation |
| Gate 3: Fairness | `fairness_passed = True` | Cohen's d ≤ 0.5 per group |

Gate results are logged as `gate_results.json` artifact to the MLflow run for full auditability.

On passing all gates:
1. Model registered to MLflow Registry as `Staging`
2. Auto-promoted to `Production` if previous production version's NDCG is lower
3. Previous `Production` archived (not deleted — rollback is always available)

**Rollback**: To restore a previous model version:
```bash
# Via MLflow UI or CLI:
mlflow models transition-model-version-stage \
  --name xgb_recipe_ranker \
  --version <old_version> \
  --stage Production
```
The serving layer detects the stage change and hot-reloads within 60 seconds — no restart required.

---

### 4.8 Production Serving

**File**: `src/serving/app_production.py`
**Container**: `sparky-serving` (profile: `serving`, `full`), port 8000

The serving layer is a FastAPI application with 2 Uvicorn workers.

#### Model Hot-Reload

A background thread runs every 60 seconds checking the MLflow Registry for a new `Production` version:

```python
with _model_lock:  # threading.RLock
    # Check MLflow Registry for latest Production version
    # If new version found: load model, rebuild explainer, update globals
    _model = new_model
    _model_version = new_version
    _explainer = Explainer(new_model, feature_columns=FEATURE_COLUMNS)
```

This means deployments require zero downtime — the old model continues serving while the new one loads.

#### `/predict` Endpoint

```
POST /predict
Request: {request_id, model_name, instances: [{47 feature keys}]}

1. assemble_features(instances)
   → 47-dim float32 matrix + user_ids + recipe_ids
2. DMatrix(features, group=[len(instances)])
3. booster.predict(dmatrix) → raw LambdaRank scores
4. Sort by score descending → assign ranks 1..N
5. prediction_logger.log_batch()     → PostgreSQL predictions table
6. prediction_logger.log_features()  → PostgreSQL inference_features table
   (feeds drift monitor with live feature distributions)
7. Return PredictResponse

Response: {request_id, model_name, model_version, generated_at,
           predictions: [{user_id, recipe_id, score, rank}]}
```

#### `/explain` Endpoint

```
POST /explain
Request: {instance: {47 feature keys}, top_k: 10}

1. Get _explainer (SHAP TreeExplainer, rebuilt after each model reload)
2. explainer.explain_prediction(instance, top_k=top_k)
   → SHAP values for this specific instance
   → Sort by |SHAP value| descending
   → Return top-K feature attributions
3. Return {feature: importance} dict for the top_k features
```

This endpoint enables users and operators to understand *why* a specific recipe was ranked highly or lowly.

---

### 4.9 Inference Feature Logging

**File**: `src/serving/prediction_logger.py`
**Called from**: `app_production.py` `/predict` handler

Every prediction request captures the full 47-feature vector to PostgreSQL:

```sql
-- Table: inference_features
CREATE TABLE inference_features (
    id           SERIAL PRIMARY KEY,
    request_id   TEXT,
    model_version TEXT,
    features     JSONB,
    captured_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

The drift monitor reads from this table to compare live feature distributions against the training baseline. Without this logging, drift detection cannot function. This was the critical missing link identified and fixed in this milestone.

---

### 4.10 Drift Monitoring and Retraining Loop

**File**: `src/data/drift_monitor.py`
**Container**: `sparky-drift-monitor` (profile: `monitoring`, `full`)
**Cycle**: Every 300 seconds (5 minutes)

The drift monitor closes the continuous learning loop:

#### KS-Test Drift Detection

For each of 19 monitored features, a **Kolmogorov-Smirnov two-sample test** is run between:
- **Baseline**: Training data distribution (from `train.csv`)
- **Live**: Recent inference feature distributions (last 24 hours from `inference_features` table)

```python
ks_stat, p_value = stats.ks_2samp(baseline_clean, live_clean)
drift_detected = p_value < DRIFT_THRESHOLD  # default: 0.05
```

**Features monitored** (19 total):
- Recipe: `minutes`, `n_ingredients`, `n_steps`, `avg_rating`, `n_reviews`
- Nutrition: `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrate`
- User targets: `daily_calorie_target`, `protein_target_g`, `carbs_target_g`, `fat_target_g`
- History: `history_pc1`, `history_pc2`, `history_pc3`

**Overall drift**: Triggered if >30% of features show individual drift (`drift_rate > 0.3`).

#### Drift Response

When overall drift is detected:
```python
trigger_retraining(reason=f"drift_detected: {drift_result['drifted_features'][:5]}")
# → POST http://retrain-api:8080/trigger
# → retrain-api launches full retrain_pipeline
```

Per-feature results are stored in PostgreSQL `drift_log` table and visible in Grafana.

#### Privacy Retention

At the start of every monitoring cycle:
```python
cleanup_old_inference_features(conn, retention_days=90)
# DELETE FROM inference_features WHERE captured_at < NOW() - INTERVAL '90 days'
```

This enforces a 90-day maximum retention on user inference data. See [Section 5.4 Privacy](#54-privacy).

---

### 4.11 CI/CD Pipeline (GitHub Actions)

**File**: `.github/workflows/retrain.yml`
**Triggers**: Push to main, weekly cron (Sundays), drift webhook

Four-stage pipeline with automatic rollback:

```
Stage 1: Data Quality Check
  → python src/data/drift_monitor.py --report-only
  → Exit non-zero if training data quality checks fail
  → Blocks training if data is bad

Stage 2: Train + Fairness Gate + Register
  → python -m src.training.retrain_pipeline --config configs/training/xgb_ranker.yaml
  → Includes Step 3.5 (fairness + SHAP) internally
  → Registers to MLflow Registry if all gates pass

Stage 3: Auto-Promote
  → python src/training/promote_model.py
  → Transitions Staging → Production if NDCG improved
  → Serving layer hot-reloads within 60 seconds

Stage 4: Rollback (on failure)
  → python src/training/rollback_model.py
  → Transitions previous Archived version back to Production
  → Triggered automatically if Stage 2 or Stage 3 fails
```

The entire workflow runs without SSH access to the VM — promotions and rollbacks are managed through MLflow's model registry API, and the serving layer polls for changes automatically.

---

## 5. Safeguarding Plan

The safeguarding plan is fully wired into the production system — not documentation-only. Every mechanism listed below is an active code path that runs in production.

### 5.1 Fairness

**Principle**: The recommendation system must not disadvantage users based on their dietary group membership.

**Implementation** (`safeguarding/fairness_checker.py`):
- After every training run (Step 3.5), the trained model's predictions on the held-out test set are analyzed across dietary groups: vegetarian, vegan, gluten-free, dairy-free, low-sodium, low-fat users
- **Cohen's d** measures the effect size between each group's score distribution and the overall population
- **Hard gate**: If any group has Cohen's d > 0.5, `fairness_passed = False` → the model is rejected and **not registered** to MLflow
- Result logged as `fairness_report.json` artifact in every MLflow training run
- Gate 3 in `model_registry.py` explicitly blocks registration on fairness failure

**Why Cohen's d**: Unlike statistical tests (which flag trivially small differences in large samples), Cohen's d measures practical significance. A score of 0.5 represents a "medium" effect — large enough to constitute meaningful unfairness in recommendation quality.

### 5.2 Explainability

**Principle**: Users and operators must be able to understand *why* a specific recipe was recommended.

**Implementation** (`safeguarding/explainability.py`):

**Training-time** (global):
- SHAP `TreeExplainer` computes mean |SHAP values| across the test set
- Result stored as `shap_global_importance.json` artifact in every MLflow run
- Enables audit of which features drive the model's decisions globally

**Serving-time** (per-instance):
- `/explain` endpoint accepts a single recipe instance
- Returns top-K features with their SHAP attributions for that specific prediction
- Example response:
  ```json
  {
    "calories": 0.42,
    "protein": 0.31,
    "avg_rating": 0.18,
    "history_pc1": 0.15,
    "carbohydrate": -0.12
  }
  ```
- `_explainer` is rebuilt from the new model on every hot-reload under `threading.RLock`

### 5.3 Transparency

**Principle**: The system's operation, data sources, and decision logic must be auditable.

**Implementation**:
- Every training run is fully logged to MLflow: all hyperparameters (flattened YAML), NDCG@10, fairness result, SHAP importances, config file, manifest hash
- `manifest.json` pins the exact dataset used (SHA-256 + git commit) — any past training run can be reproduced exactly
- Gate results (`gate_results.json`) logged as artifact — any model in the registry has a full audit trail
- Prometheus metrics expose inference volume, latency, and error rates at `/metrics`
- Grafana dashboards make drift trends, retraining history, and serving performance visible to operators
- `drift_log` PostgreSQL table records every KS-test result with timestamp
- All source code is in a public GitHub repository

### 5.4 Privacy

**Principle**: User feature data captured at inference time must not be retained indefinitely.

**Implementation** (`src/data/drift_monitor.py`):
```python
def cleanup_old_inference_features(conn, retention_days: int = 90) -> int:
    # DELETE FROM inference_features WHERE captured_at < NOW() - INTERVAL '90 days'
```

- Called at the start of **every** monitoring cycle (every 300 seconds)
- Enforces a 90-day maximum retention window on all inference feature logs
- The `inference_features` table stores feature vectors (not raw user PII) but still contains user nutritional targets and dietary preferences — hence the retention policy
- Deletion is logged: `"Privacy retention: deleted N rows older than 90 days"`
- Non-fatal: if the cleanup query fails (e.g., DB connectivity), the monitoring cycle continues

**Additional privacy measures**:
- Allergen flags and dietary restrictions are stored as numerical features, not free text
- `user_id` is an integer (anonymous ID) — no email, name, or contact information stored in the ML system
- PostgreSQL `user_interactions` and `inference_features` tables are not exposed externally

### 5.5 Accountability

**Principle**: Every model deployed to production must have a traceable decision trail.

**Implementation**:
- **MLflow Registry**: Every model version has a state machine (None → Staging → Production → Archived). Transitions are timestamped and attributed.
- **Gate results artifact**: Every registered model has `gate_results.json` containing all three gate verdicts (NDCG, overfit, fairness).
- **Retraining triggers**: `drift_monitor.py` logs the reason for every retraining trigger (`drift_detected: [feature_list]`). GitHub Actions logs every workflow run with trigger reason.
- **`retrain-api` (port 8080)**: All retraining requests go through this API — creates an audit trail of who/what triggered retraining and when.
- **`rollback_model.py`**: Rollback events are explicit MLflow stage transitions, visible in the registry history.
- **Prometheus + Grafana**: Operator-visible dashboards for model behavior over time.

### 5.6 Robustness

**Principle**: The system must handle failures gracefully and degrade predictably rather than catastrophically.

**Implementation**:

| Mechanism | Description |
|---|---|
| **Safeguarding fallback** | All imports of `safeguarding.*` wrapped in `try/except` — missing `shap` package logs a warning and disables the feature, does not crash training/serving |
| **Model fallback** | `MODEL_FALLBACK_PATH=/models/xgboost_ranker.json` — if MLflow is unavailable, serving loads the last-known-good model from disk |
| **Hot-reload under lock** | `threading.RLock` prevents race conditions during model hot-reload — in-flight requests complete against the old model |
| **Drift monitor loop** | Exceptions in the monitoring cycle are caught and logged; the loop continues after `CHECK_INTERVAL_SECONDS` |
| **Drift webhook retry** | `requests.post()` has 10-second timeout; failure is logged but non-fatal — the monitor continues |
| **DB connection resilience** | All DB operations have `try/except` with `conn.rollback()` on failure |
| **MLflow healthcheck** | Docker Compose healthcheck on `mlflow:5000/health` before training or serving containers start |
| **Postgres healthcheck** | `pg_isready -U sparky -d sparky` healthcheck with 5 retries |
| **CI/CD rollback** | GitHub Actions Stage 4 automatically rolls back to the previous Production model if training or promotion fails |
| **Time-stratified splits** | Temporal train/val/test split prevents future data from contaminating training — a robustness guarantee against evaluation inflation |

---

## 6. SparkyFitness Integration

SparkyFitness is the open-source nutrition tracking application that this ML system integrates with. The ML serving layer is wired into the SparkyFitness food/recipe recommendation flow.

### Integration Architecture

```
SparkyFitness App
    ↓ User visits "Recommended Recipes" page
    ↓ App calls internal recommendation service
    ↓ ML_RECOMMENDATION_URL=http://sparky-serving:8000
    ↓ POST /predict with user features + candidate recipes
    ↓ SparkyFitness renders ranked recipe list to user
```

### Network Configuration

Both the SparkyFitness application and the ML system share `sparky-net` (Docker bridge network):

```yaml
# In SparkyFitness docker-compose.yml:
networks:
  sparky-net:
    external: true

# Environment:
ML_RECOMMENDATION_URL: http://sparky-serving:8000
```

### What the Integration Does

1. SparkyFitness assembles candidate recipes for a user from its recipe database
2. For each candidate, it constructs a feature dict matching the 47-feature schema
3. Sends `POST /predict` with all candidates as a batch
4. The ML model scores and ranks the candidates
5. SparkyFitness renders the ranked list — the ML model's ranking replaces the default sort order
6. User interactions (clicks, cooks, skips) are logged back to PostgreSQL `user_interactions`
7. These interactions feed the next retraining cycle

The integration means ML recommendations appear in the **regular user flow** of the SparkyFitness application, not as a separate feature.

---

## 7. Feature Schema (47 Dimensions)

All features are encoded as `float32`. Missing values are filled with `0.0`. String/categorical features use sorted-unique integer mapping (unknown → -1).

| Category | Features | Count |
|---|---|---|
| **Recipe basics** | `minutes`, `n_ingredients`, `n_steps`, `avg_rating`, `n_reviews` | 5 |
| **Cuisine** | `cuisine` (string → integer encoding) | 1 |
| **Nutrition** | `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrate` | 7 |
| **Allergens** | `has_egg`, `has_fish`, `has_milk`, `has_nuts`, `has_peanut`, `has_sesame`, `has_shellfish`, `has_soy`, `has_wheat` | 9 |
| **User macro targets** | `daily_calorie_target`, `protein_target_g`, `carbs_target_g`, `fat_target_g` | 4 |
| **User dietary flags** | `user_vegetarian`, `user_vegan`, `user_gluten_free`, `user_dairy_free`, `user_low_sodium`, `user_low_fat` | 6 |
| **User history PCA** | `history_pc1`, `history_pc2`, `history_pc3` | 3 |
| **Nutrition (g variants)** | `calories_g`, `total_fat_g`, `sugar_g`, `sodium_g`, `protein_g`, `saturated_fat_g`, `carbohydrate_g` | 7 |
| **Interaction context** | `meal_type`, `time_of_day`, `day_of_week`, `interaction_count`, `user_avg_rating` | 5 |
| **Total** | | **47** |

The training pipeline (`ranking_data.py`) and serving pipeline (`assemble_features()` in `app_production.py`) use the same ordered feature list — eliminating train/serve skew.

---

## 8. API Contracts

### GET /health

```json
{
  "status": "healthy",
  "model": "xgb_recipe_ranker",
  "model_version": "3",
  "backend": "xgboost_production",
  "uptime_seconds": 3642
}
```

### POST /predict

**Request**:
```json
{
  "request_id": "req_20260416_001",
  "model_name": "xgb_ranker",
  "instances": [
    {
      "user_id": 37449,
      "recipe_id": 33096,
      "minutes": 10,
      "cuisine": "italian",
      "n_ingredients": 8,
      "n_steps": 4,
      "avg_rating": 4.5,
      "n_reviews": 120,
      "calories": 380.0,
      "protein": 22.0,
      "total_fat": 14.0,
      "carbohydrate": 45.0,
      "sugar": 5.0,
      "sodium": 620.0,
      "saturated_fat": 4.0,
      "has_milk": 1, "has_egg": 0, "has_nuts": 0, "has_peanut": 0,
      "has_fish": 0, "has_shellfish": 0, "has_wheat": 1, "has_soy": 0, "has_sesame": 0,
      "daily_calorie_target": 2200.0,
      "protein_target_g": 60.0,
      "carbs_target_g": 275.0,
      "fat_target_g": 65.0,
      "user_vegetarian": 0, "user_vegan": 0, "user_gluten_free": 0,
      "user_dairy_free": 0, "user_low_sodium": 0, "user_low_fat": 0,
      "history_pc1": 0.42, "history_pc2": -0.11, "history_pc3": 0.07
    }
  ]
}
```

**Response**:
```json
{
  "request_id": "req_20260416_001",
  "model_name": "xgb_ranker",
  "model_version": "3",
  "generated_at": "2026-04-16T14:30:00Z",
  "predictions": [
    {"user_id": 37449, "recipe_id": 33096, "score": 1.8421, "rank": 1},
    {"user_id": 37449, "recipe_id": 120964, "score": 0.9174, "rank": 2}
  ]
}
```

### POST /explain

**Request**:
```json
{
  "instance": {
    "user_id": 37449,
    "recipe_id": 33096,
    "calories": 380.0,
    "protein": 22.0,
    "...": "..."
  },
  "top_k": 10
}
```

**Response**:
```json
{
  "calories": 0.42,
  "protein": 0.31,
  "avg_rating": 0.18,
  "history_pc1": 0.15,
  "carbohydrate": -0.12,
  "daily_calorie_target": 0.09,
  "has_milk": 0.07,
  "n_steps": -0.05,
  "sugar": -0.04,
  "total_fat": 0.03
}
```

Positive values indicate features that pushed the score up (more likely to be recommended). Negative values indicate features that pulled the score down.

### GET /metrics

Prometheus scrape endpoint. Key metrics:

```
http_requests_total{method="POST", endpoint="/predict", status="200"} 14823
http_request_duration_seconds_p50{endpoint="/predict"} 0.041
http_request_duration_seconds_p99{endpoint="/predict"} 0.087
model_version{version="3"} 1
inference_features_logged_total 14823
```

---

## 9. Infrastructure and Container Architecture

### Single Multi-Stage Dockerfile

All services are built from a single `Dockerfile` with four named stages. Each stage inherits the shared `base` stage which includes the `safeguarding/` module:

```
base (python:3.11-slim + gcc + libpq-dev + safeguarding/)
├── data     → data pipeline containers (batch-pipeline, data-generator, drift-monitor)
├── training → trainer + retrain-api containers
└── serving  → sparky-serving container

mlflow (python:3.11-slim + mlflow==3.1.0, standalone — no base needed)
```

The `base` stage's inclusion of `safeguarding/` is critical: both the `training` and `serving` stages import from `safeguarding.*`. Previously (with 4 separate Dockerfiles), neither had access to this module, causing runtime import errors.

### Twelve Containers

| Container | Image/Stage | Profile | Port | Purpose |
|---|---|---|---|---|
| `sparky-postgres` | postgres:15-alpine | always | 5433 | PostgreSQL database |
| `sparky-mlflow` | mlflow stage | always | 5000 | Experiment tracking + model registry |
| `sparky-data-generator` | data stage | data, full | — | Synthetic interaction replay |
| `sparky-batch-pipeline` | data stage | data, retrain, full | — | Dataset compilation |
| `sparky-drift-monitor` | data stage | monitoring, full | — | KS-test drift detection |
| `sparky-trainer` | training stage | retrain, full | — | Model training (one-shot) |
| `sparky-retrain-api` | training stage | monitoring, full | 8080 | Retraining webhook receiver |
| `sparky-serving` | serving stage | serving, full | 8000 | Prediction + explain API |
| `sparky-prometheus` | prom/prometheus | monitoring, full | 9090 | Metrics collection |
| `sparky-alertmanager` | prom/alertmanager | monitoring, full | 9093 | Alert routing |
| `sparky-grafana` | grafana/grafana | monitoring, full | 3000 | Visualization dashboards |
| `sparky-postgres-exporter` | postgres-exporter | monitoring, full | 9187 | DB metrics for Prometheus |

### Running the System

```bash
# Full system (all containers)
make run-all
# or: docker compose --profile full up -d

# Serving only (inference + MLflow + Postgres)
docker compose --profile serving up -d

# Monitoring stack
docker compose --profile monitoring up -d

# One-shot training run
docker compose --profile retrain run --rm trainer
```

### Chameleon Cloud Deployment

- **Compute**: KVM@TACC VM (Ubuntu 22.04), minimum 4 vCPU / 8 GB RAM
- **Storage**: Chameleon OpenStack Swift (CHI@TACC) for versioned training datasets
- **Lease**: 7-day renewable Chameleon lease
- **Access**: SSH key pair, floating IP assigned
- **Ports opened**: 8000 (serving), 5000 (MLflow), 3000 (Grafana), 9090 (Prometheus)

---

## 10. Monitoring Stack

### Prometheus Alert Rules (`monitoring/alert_rules.yml`)

| Alert | Condition | Severity |
|---|---|---|
| `HighErrorRate` | Error rate > 5% for 5 min | critical |
| `HighLatency` | P95 latency > 500 ms for 10 min | warning |
| `ModelStale` | No inference in 30 min (model not loaded) | warning |
| `DriftDetected` | Drift log shows overall_drift_detected = true | warning |
| `RetrainingFailed` | CI/CD workflow failed | critical |

### Grafana Dashboards (`monitoring/grafana/dashboards/`)

**Inference Dashboard**:
- Request rate (req/sec)
- P50/P95/P99 latency histograms
- Error rate by endpoint
- Model version currently serving

**Drift Dashboard**:
- KS statistic per feature over time (from `drift_log` table)
- Features in drift (time series)
- Retraining trigger events

**Training Dashboard**:
- NDCG@10 per training run over time
- Gate pass/fail history
- Fairness score per dietary group per run

---

## 11. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Data Processing** | Pandas | 2.2.0+ | ETL, feature engineering |
| | NumPy | 1.26.0+ | Numerical operations |
| | Scikit-learn | 1.3+ | PCA, preprocessing |
| | SciPy | 1.13+ | KS-test (drift monitor) |
| **ML Training** | XGBoost | 2.1.4 | LambdaRank model |
| **Explainability** | SHAP | 0.45+ | TreeExplainer, feature attribution |
| **Experiment Tracking** | MLflow | 3.1.0 | Run tracking, model registry |
| **API Server** | FastAPI | 0.110.0+ | REST inference endpoints |
| | Uvicorn | 0.29.0+ | ASGI server |
| | Pydantic | 2.0+ | Request/response validation |
| **Databases** | PostgreSQL | 15 | User interactions, features, drift log |
| | psycopg2 | 2.9+ | Direct PostgreSQL queries |
| **Object Storage** | Chameleon OpenStack Swift | — | Versioned dataset storage |
| **Data Quality** | Soda Core | — | Automated validation checks |
| **Monitoring** | Prometheus | 2.51.2 | Metrics scraping |
| | Grafana | 10.4.2 | Dashboards |
| | Alertmanager | 0.27.0 | Alert routing |
| **Containerization** | Docker | — | Multi-stage builds |
| | Docker Compose | v3.9 | Multi-service orchestration |
| **CI/CD** | GitHub Actions | — | 4-stage automated pipeline |
| **Compute** | Chameleon Cloud (KVM@TACC) | — | VM-based training/serving |
| **Languages** | Python | 3.11 | All components |
| **Config Format** | YAML | — | Training hyperparameters |

---

## 12. Evaluation and Metrics

### Primary Metric: NDCG@10

Normalized Discounted Cumulative Gain at rank 10. Measures ranking quality penalizing relevant items placed lower in the list.

```
DCG@10  = Σ(rel_i / log2(i+1))  for i = 1..10
IDCG@10 = DCG@10 with ideal ordering
NDCG@10 = DCG@10 / IDCG@10

Final NDCG@10 = mean(NDCG@10 per user)
```

**Achieved**: NDCG@10 = **0.8148** on held-out test set (substantially above 0.55 gate threshold).

### Quality Gates Summary

| Gate | Metric | Threshold | Current Value |
|---|---|---|---|
| Performance | NDCG@10 | ≥ 0.55 | **0.8148** |
| Overfit | test_loss / val_loss | ≤ 1.10 | Within bounds |
| Fairness | Cohen's d (dietary groups) | ≤ 0.5 | Within bounds |

### Latency Targets

| Endpoint | P50 | P99 |
|---|---|---|
| `/predict` | < 50 ms | < 100 ms |
| `/explain` | < 200 ms | < 500 ms |
| `/health` | < 5 ms | < 20 ms |

---

## 13. Data Strategy

### Source Data

| Dataset | Source | Size | Description |
|---|---|---|---|
| `RAW_recipes.csv` | Kaggle Food.com | ~230K rows | Recipes with nutrition, tags, ingredients |
| `RAW_interactions.csv` | Kaggle Food.com | ~1.1M rows | User-recipe ratings (1–5 stars) |

### Temporal Split Strategy

To prevent label leakage, all splits are time-based:

```
Timeline ───────────────────────────────────────────────────────►
                        ▲              ▲              ▲
                 test_start        val_start       last_date
                 (−14 days)        (−7 days)

TRAIN: All interactions before test_start
TEST:  test_start → val_start  (7-day window)
VAL:   val_start  → last_date  (7-day window)
```

### Label Definition

- **Positive** (`label=1`): `rating ≥ 4` — user cooked and enjoyed the recipe
- **Negative** (`label=0`): `rating < 4` — user did not find the recipe satisfying
- Expected positive rate: ~40–60% (validated by Soda checks)

### Versioning and Reproducibility

Full provenance chain for any trained model:
```
model artifact
    → MLflow run_id
    → gate_results.json (NDCG, overfit, fairness verdicts)
    → shap_global_importance.json (feature importance snapshot)
    → fairness_report.json (per-group Cohen's d values)
    → manifest.json
    → train.sha256 + val.sha256 + test.sha256
    → batch_pipeline Git commit hash
    → PostgreSQL snapshot at manifest timestamp
```

Given any registered model, the exact dataset, code, and hyperparameters used to train it can be recovered.

---

## 14. Deliverables

### Joint Deliverables (12/15 points) — COMPLETED

- [x] **End-to-end automated workflow**: Data generation → batch compilation → training → fairness gate → MLflow registration → serving hot-reload → drift monitoring → retraining webhook. Runs via GitHub Actions and docker-compose without manual SSH intervention.
- [x] **Production deployment on Chameleon Cloud**: All 12 containers deployed on KVM@TACC VM. Accessible at the VM's floating IP. Described in `CHAMELEON_DEPLOYMENT.md`.
- [x] **SparkyFitness integration**: ML recommendations wired into the SparkyFitness regular user flow via `ML_RECOMMENDATION_URL` and shared `sparky-net` Docker network. Calling `POST /predict` for every recipe recommendation in the app.
- [x] **Safeguarding plan implemented within the system**: Fairness gate (Cohen's d, blocking), SHAP explainability (training artifact + `/explain` endpoint), 90-day privacy retention, MLflow audit trail, Prometheus/Grafana transparency, KS-test robustness. All 6 principles have concrete code mechanisms. See [Section 5](#5-safeguarding-plan).
- [x] **Minimal human intervention**: Promotions happen via MLflow Registry API transitions detected by the serving layer's hot-reload loop. Rollbacks triggered automatically by GitHub Actions on CI failure. Manual approval available but not required.

### Data Team Deliverables (3/15 points) — COMPLETED

- [x] **Data pipeline with ingestion quality checks** (`run_soda_checks.py`, `check_ingestion_quality()` in `drift_monitor.py`): Schema validation, null checks, row count thresholds, calorie range validation
- [x] **Training set quality checks** (`check_training_set_quality()`, `batch_pipeline.py`): Label balance, zero-variance column detection, temporal ordering validation, no-null checks on critical columns
- [x] **Live inference drift monitoring** (`drift_monitor.py`): KS-test on 19 features every 300 seconds, drift rate threshold (>30%), automatic retraining trigger, results logged to PostgreSQL `drift_log` table
- [x] **Versioned dataset storage** (Chameleon Swift + `manifest.json`): SHA-256 content hashes, git commit pinning, full lineage chain
- [x] **Synthetic data generation** (`data_generator.py`): Realistic user profiles, allergen constraints, chronological interaction replay, continuous PostgreSQL population
- [x] **90-day inference feature retention** (`cleanup_old_inference_features()`): Privacy-compliant automatic deletion

### Training Team Deliverables (3/15 points) — COMPLETED

- [x] **XGBoost LambdaRank model** (`retrain_pipeline.py`): `rank:ndcg` objective, NDCG@10 = 0.8148, 47-feature vector, early stopping
- [x] **MLflow experiment tracking**: All params + metrics + artifacts logged per run; model registry with Staging/Production/Archived stages
- [x] **Automated retraining pipeline**: `retrain_pipeline.py` triggered by: CI/CD push, weekly cron, drift webhook (POST to `retrain-api`)
- [x] **Fairness gate wired into training** (Step 3.5): Cohen's d per dietary group, hard blocker on registration if fairness_passed = False
- [x] **SHAP explainability at training time**: `shap_global_importance.json` artifact logged to every MLflow run
- [x] **Quality gates**: 3-gate system (NDCG threshold + overfit check + fairness) before MLflow registration
- [x] **Auto-promote and rollback**: GitHub Actions Stage 3 (promote) + Stage 4 (rollback on failure)
- [x] **YAML-driven hyperparameter config** (`configs/training/xgb_ranker.yaml`): Structured, version-controlled config

### Serving Team Deliverables (3/15 points) — COMPLETED

- [x] **Production FastAPI serving** (`app_production.py`): 2 Uvicorn workers, `/health`, `/predict`, `/explain`, `/metrics` endpoints
- [x] **Zero-downtime model hot-reload**: Background thread polls MLflow Registry every 60 seconds; `threading.RLock` prevents race conditions
- [x] **SHAP /explain endpoint**: Per-instance SHAP attribution, `_explainer` rebuilt on every model reload
- [x] **Inference feature logging**: `prediction_logger.log_features()` called on every `/predict` request; populates `inference_features` table for drift monitor
- [x] **Prometheus metrics**: `http_requests_total`, latency histograms, model version gauge at `/metrics`
- [x] **Grafana dashboards**: Inference latency, drift trends, training history
- [x] **Alert rules**: High error rate, latency, model staleness, drift detection
- [x] **Single Dockerfile**: Multi-stage build with shared `base` carrying `safeguarding/`; all 4 service types built from one file
