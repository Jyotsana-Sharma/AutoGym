# Project Proposal: SparkyFitness — Personalized Recipe Recommendation System

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Component Breakdown](#5-component-breakdown)
   - 5.1 [Sparky Data Pipeline](#51-sparky-data-pipeline)
   - 5.2 [AutoGym-Main: Model Training](#52-autogym-main-model-training)
   - 5.3 [AutoGym-Serving: Model Inference](#53-autogym-serving-model-inference)
6. [Data Strategy](#6-data-strategy)
7. [Machine Learning Design](#7-machine-learning-design)
8. [Infrastructure & DevOps](#8-infrastructure--devops)
9. [API Contracts](#9-api-contracts)
10. [Evaluation & Metrics](#10-evaluation--metrics)
11. [Scalability & Performance](#11-scalability--performance)
12. [MLOps & Continuous Learning](#12-mlops--continuous-learning)
13. [Technology Stack Summary](#13-technology-stack-summary)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [Deliverables](#15-deliverables)

---

## 1. Executive Summary

**SparkyFitness** is a production-grade, end-to-end personalized recipe recommendation system that ranks meal suggestions according to each user's nutritional goals, dietary restrictions, allergen constraints, and personal cooking history.

The system is composed of three tightly integrated sub-projects:

| Sub-Project | Role |
|---|---|
| `sparky-data-pipeline` | Raw data ingestion, feature engineering, data quality, versioned dataset production |
| `AutoGym-main` | XGBoost LambdaRank model training, hyperparameter management, MLflow experiment tracking |
| `AutoGym-serving` | Production REST API serving (multiple backends), Prometheus monitoring, benchmarking |

Together, they form a **continuous learning flywheel**: user interactions feed back into the pipeline, trigger batch retraining, and deploy updated models to production — all with full dataset reproducibility and temporal correctness guarantees.

---

## 2. Problem Statement

Generic recipe recommendation systems suffer from two fundamental limitations:

1. **Context Blindness**: They do not account for individual nutritional targets, allergen sensitivities, or cooking time constraints.
2. **Static Personalization**: They rely on collaborative filtering signals (e.g., popularity) without learning from a user's evolving cooking preferences over time.

The result is recommendations that are either nutritionally inappropriate, potentially unsafe (allergens), or simply irrelevant to what a user can cook today.

**SparkyFitness** addresses this by framing recipe recommendation as a **supervised learning-to-rank problem**, where each candidate recipe is scored against the full context of the user's profile, producing a personalized ranking that is nutritionally aware, allergen-safe, and preference-aligned.

---

## 3. Proposed Solution

### Core Approach

- **Learning to Rank** using XGBoost LambdaRank (`rank:ndcg` objective).
- Each (user, recipe) pair is represented as a 44-dimensional feature vector combining recipe attributes, user nutritional targets, allergen flags, and PCA-encoded cooking history.
- At inference time, a ranked list of candidate recipes is returned for each user in under 100 ms.

### Key Differentiators

| Feature | Description |
|---|---|
| Nutritional Targeting | User macro targets (calories, protein, carbs, fat) are first-class features |
| Allergen Safety | 9 allergen flags (milk, egg, nuts, peanut, fish, shellfish, wheat, soy, sesame) are both features and hard rule filters |
| PCA Cooking History | 6-dimensional PCA embedding of the user's cuisine cooking history |
| Temporal Correctness | Time-stratified train/val/test splits prevent data leakage |
| Reproducibility | SHA-256 dataset hashes + Git commit pinned in every training manifest |
| Multi-backend Serving | Baseline XGBoost, optimized (LRU cache + buffer pooling), ONNX Runtime, NVIDIA Triton |
| Continuous Retraining | Data generator simulates production user interactions; batch pipeline merges with historical data |

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPARKY DATA PIPELINE                                 │
│                                                                             │
│  [Kaggle Food.com Dataset]                                                  │
│         ↓                                                                   │
│  build_training_table.py                                                    │
│    • Cuisine mapping (100+ tags → 20 categories)                            │
│    • Allergen detection (keyword scanning)                                  │
│    • Nutritional parsing (macro columns)                                    │
│    • Label generation (rating ≥ 4 → positive)                              │
│    • PCA cooking history embeddings (6 components)                          │
│    • Time-stratified split (train / val / test)                             │
│         ↓                                                                   │
│  ingest_to_object_store.py                                                  │
│    • SHA-256 hash versioning                                                │
│    • Gaussian noise expansion (simulates large-scale dataset)               │
│    • Upload to Chameleon OpenStack Swift                                    │
│         ↓                                                                   │
│  data_generator.py → PostgreSQL (user_interactions)                         │
│    • Synthetic user profile simulation                                      │
│    • Replay Kaggle interactions + simulated actions                         │
│         ↓                                                                   │
│  batch_pipeline.py                                                          │
│    • Merge historical CSVs + new PostgreSQL interactions                    │
│    • Generate versioned manifest.json (hashes + Git commit)                 │
│    • Upload to Swift: proj04-sparky-training-data                           │
│         ↓                                                                   │
│  run_soda_checks.py  ← Data quality gates (Soda Core)                      │
│                                                                             │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                   train.csv / val.csv / test.csv
                         manifest.json
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTOGYM-MAIN (TRAINING)                             │
│                                                                             │
│  configs/train/*.yaml → train.py                                            │
│    • XGBoost LambdaRank (rank:ndcg)                                         │
│    • Optional: Ray distributed training (4 workers)                         │
│    • Evaluation: NDCG@10                                                    │
│    • MLflow: param logging, artifact storage, run comparison                │
│                                                                             │
│  Output: Trained model (JSON) + ONNX conversion                             │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                        Trained Model
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AUTOGYM-SERVING (INFERENCE)                            │
│                                                                             │
│  FastAPI REST API (/health, /predict)                                       │
│    ├─ app_baseline.py     — Raw XGBoost                                     │
│    ├─ app_optimized.py    — LRU cache + pre-allocated buffers               │
│    ├─ app_onnx.py         — ONNX Runtime (CPU-optimized)                    │
│    └─ Triton backend      — NVIDIA GPU inference                            │
│                                                                             │
│  Feature assembly (44-dim) → DMatrix → ranked predictions                  │
│  Prometheus metrics: request count, latency, error rate                     │
│  Benchmarking: scripts/benchmark.py                                         │
│                                                                             │
│  Output: PredictResponse {request_id, predictions[{user_id, recipe_id,     │
│          score, rank}]}                                                      │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                    User Interactions Logged
                              │
                              ↓
                     PostgreSQL (user_interactions)
                              │
                   [Batch pipeline picks up new data]
                              │
                    ← Continuous Learning Flywheel →
```

---

## 5. Component Breakdown

### 5.1 Sparky Data Pipeline

**Repository**: `sparky-data-pipeline/`

#### 5.1.1 Core Feature Engineering (`build_training_table.py`)

The central ETL script transforms raw Kaggle Food.com data into model-ready training tables.

**Input Data Sources**:
- `RAW_recipes.csv` — ~230K recipes with ingredients, nutrition, tags, steps
- `RAW_interactions.csv` — ~1.1M user-recipe interactions with ratings

**Processing Pipeline**:

| Step | Module | Description |
|---|---|---|
| Cuisine Mapping | Tag normalization | 100+ Kaggle tags → 20 canonical cuisines (italian, american, asian, mexican, etc.) |
| Allergen Detection | Keyword scan | Scans `ingredients` and `tags` for 9 allergens; outputs binary flags |
| Nutrition Parsing | String → floats | Parses `[calories, total_fat, sugar, sodium, protein, saturated_fat, carbohydrate]` |
| Aggregation | Recipe stats | `avg_rating`, `n_reviews` from interaction join |
| Filtering | Quality gates | `minutes ≤ 99th percentile`, `n_reviews ≥ 3` |
| Label Generation | Binary | `rating ≥ 4 → label=1` (positive interaction) |
| PCA Embeddings | Cooking history | One-hot cuisine vectors → PCA(n=6) → `history_pc1...history_pc6` |
| Time Split | Temporal | Train: all except last 14d; Val: last 7d; Test: preceding 7d |

**Key Configuration Constants**:
```python
POSITIVE_RATING_THRESHOLD = 4    # >= 4 → positive label
MIN_USER_INTERACTIONS     = 5    # Drop sparse users
HISTORY_PCA_COMPONENTS    = 6    # Latent cooking preference dims
MINUTES_QUANTILE_UPPER    = 0.99 # Cap extreme cooking times
```

**Outputs**: `enriched_recipes.csv`, `training_table.csv`, `train.csv`, `val.csv`, `test.csv`

---

#### 5.1.2 Data Ingestion & Versioning (`ingest_to_object_store.py`)

Uploads raw data to Chameleon OpenStack Swift with cryptographic versioning.

- **Schema Validation**: Enforces required columns for both recipes and interactions files before upload.
- **SHA-256 Hashing**: Computes content hashes of both raw CSVs; stored as Swift object metadata.
- **Synthetic Expansion**: If dataset is below 5 GB, applies Gaussian noise (σ=0.05) to numeric columns and doubles the dataset (EXPAND_FACTOR=2) to simulate production-scale data.
- **Object Storage**: Target container `proj04-sparky-raw-data` on Chameleon Cloud (CHI@TACC region). Container is set to public read for easy data access.

---

#### 5.1.3 Batch Training Dataset Compiler (`batch_pipeline.py`)

Merges historical datasets with fresh production interactions to build versioned training packages.

**Workflow**:
1. Load historical train/val/test CSVs from Swift.
2. Query PostgreSQL `user_interactions` table for new production interactions since last run.
3. Merge and re-split.
4. Run quality checks:
   - No empty splits
   - `train_max_date < test_min_date` (no temporal leakage)
   - Positive label rate between 10–98%
   - No nulls in `user_id`, `recipe_id`, `label`
   - Feature-label Pearson correlation < 0.9
5. Generate and upload `manifest.json`.

**Manifest Schema**:
```json
{
  "pipeline_version": "1.0.0",
  "git_commit": "<SHA>",
  "timestamp": "<ISO-8601>",
  "splits": {
    "train": { "path": "...", "sha256": "...", "rows": 100000 },
    "val":   { "path": "...", "sha256": "...", "rows": 10000  },
    "test":  { "path": "...", "sha256": "...", "rows": 10000  }
  }
}
```

---

#### 5.1.4 Online Feature Pipeline (`online_feature_pipeline.py`)

Real-time feature computation at inference time — designed for sub-100 ms end-to-end latency.

**Class: `OnlineFeaturePipeline`**

| Method | Description |
|---|---|
| `get_user_features(user_id)` | Returns macro targets; checks memory cache → PostgreSQL cache table → historical interactions → defaults |
| `rule_filter(context)` | Hard removes allergen-flagged and calorie-out-of-range recipes; tracks filter latency |
| `compute_features(user_id, context)` | Assembles 44-dim feature matrix; adds meal_type and time_of_day context features |

**Timing Breakdown Target** (< 100 ms total):
- `user_features_ms`: user preference lookup
- `rule_filter_ms`: allergen + calorie filtering
- `matrix_build_ms`: feature matrix construction

---

#### 5.1.5 Synthetic Data Generator (`data_generator.py`)

Simulates a production user base to enable offline retraining cycles without live traffic.

- Generates realistic user profiles with random allergens, dietary restrictions, and macro targets.
- Assigns meal contexts based on hour of day (breakfast 5–10am, lunch 10–2pm, etc.).
- Maps ratings to actions: `rating ≥ 4 → cook`, `3–4 → cook/view/view`, `2–3 → view/skip/skip`, `< 2 → skip`.
- Replays held-out Kaggle test interactions in chronological order at a configurable rate (default: 50 interactions/minute).
- Writes all events to PostgreSQL `user_interactions`.

---

#### 5.1.6 Data Quality Checks (`run_soda_checks.py`)

Automated data validation using **Soda Core** as a quality gate before model training.

**Checks on `enriched_recipes`**:
- `row_count > 50000`
- `missing_count(recipe_id) = 0`
- `duplicate_count(recipe_id) = 0`
- `min(calories) >= 0` and `max(calories) < 500000`
- `avg(calories) between 200 and 800`
- `min(n_reviews) >= 3`

**Checks on training splits**:
- Row count thresholds per split
- Zero nulls in critical columns
- Label balance: `avg(label) between 0.1 and 0.98`

---

#### 5.1.7 PostgreSQL Database Schema

```sql
-- Recipes master table
CREATE TABLE recipes (
  recipe_id       INTEGER PRIMARY KEY,
  name            TEXT,
  minutes         INTEGER,
  cuisine         VARCHAR(50),
  calories        REAL, protein_g REAL, carbohydrate_g REAL,
  total_fat_g     REAL, sugar_g REAL, sodium_mg REAL, saturated_fat_g REAL,
  n_ingredients   INTEGER, n_steps INTEGER,
  avg_rating      REAL, n_reviews INTEGER,
  -- Allergen flags
  has_milk SMALLINT, has_egg SMALLINT, has_nuts SMALLINT, has_peanut SMALLINT,
  has_fish SMALLINT, has_shellfish SMALLINT, has_wheat SMALLINT,
  has_soy SMALLINT, has_sesame SMALLINT,
  created_at      TIMESTAMP DEFAULT NOW()
);

-- User profiles
CREATE TABLE users (
  user_id               INTEGER PRIMARY KEY,
  dietary_restrictions  JSONB,
  macro_targets         JSONB,
  allergens             JSONB,
  created_at            TIMESTAMP DEFAULT NOW(),
  updated_at            TIMESTAMP DEFAULT NOW()
);

-- Live interaction stream
CREATE TABLE user_interactions (
  id          SERIAL PRIMARY KEY,
  user_id     INTEGER NOT NULL,
  recipe_id   INTEGER NOT NULL,
  rating      SMALLINT CHECK (rating BETWEEN 1 AND 5),
  action      VARCHAR(10) CHECK (action IN ('view', 'cook', 'rate', 'skip')),
  created_at  TIMESTAMP DEFAULT NOW()
);
-- Indexes: (user_id), (recipe_id), (created_at), (user_id, created_at)

-- Online inference cache
CREATE TABLE user_features_cache (
  user_id                INTEGER PRIMARY KEY,
  daily_calorie_target   REAL, protein_target_g REAL,
  carbs_target_g         REAL, fat_target_g REAL,
  history_pc1 REAL, history_pc2 REAL, history_pc3 REAL,
  history_pc4 REAL, history_pc5 REAL, history_pc6 REAL,
  version_id             VARCHAR(64),
  updated_at             TIMESTAMP DEFAULT NOW()
);
```

---

### 5.2 AutoGym-Main: Model Training

**Repository**: `AutoGym-main/`

#### 5.2.1 Feature Schema (44 Columns)

| Category | Features |
|---|---|
| Recipe basics (5) | `minutes`, `n_ingredients`, `n_steps`, `avg_rating`, `n_reviews` |
| Cuisine (1) | `cuisine` (string → sorted-integer encoding) |
| Nutrition (8) | `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrate` + `_g` variants |
| Allergens (9) | `has_egg`, `has_fish`, `has_milk`, `has_nuts`, `has_peanut`, `has_sesame`, `has_shellfish`, `has_soy`, `has_wheat` |
| User targets (4) | `daily_calorie_target`, `protein_target_g`, `carbs_target_g`, `fat_target_g` |
| Dietary flags (6) | `user_vegetarian`, `user_vegan`, `user_gluten_free`, `user_dairy_free`, `user_low_sodium`, `user_low_fat` |
| History PCA (6) | `history_pc1`, `history_pc2`, `history_pc3`, `history_pc4`, `history_pc5`, `history_pc6` |

All features encoded as `float32`. String/categorical columns → sorted-unique → integer mapping (unknown → -1).

---

#### 5.2.2 Training Backends

**Single-Node XGBoost** (`train.py` + `ranking_data.py`):
- Loads `PreparedFrames` dataclass (feature matrix, feature names, group arrays).
- Trains `XGBRanker` with `rank:ndcg` objective and `ndcg@10` eval metric.
- Evaluates on test set; logs all params and metrics to MLflow.
- Output: saved model artifact + JSON summary.

**Distributed Ray Training** (`ray_training.py`):
- Shards users across N workers using modulo hash on `user_id`.
- Each worker trains a local XGBoost model; aggregates results.
- Checkpointing every 25 rounds; tolerates 1 worker failure (`max_failures=1`).
- Optional worker failure injection for resilience testing.

**Baseline Popularity** (popularity ranking):
- Ranks by total positive ratings per recipe.
- Serves as a reproducible baseline for NDCG comparison.

---

#### 5.2.3 Hyperparameter Configuration (YAML-driven)

```yaml
# configs/train/xgb_ranker.yaml
run_name:        xgb_ranker_v1
experiment_name: autogym-recipe-ranking
data:
  train_path:      "train.csv"
  validation_path: "val.csv"
  test_path:       "test.csv"
training:
  objective:        rank:ndcg
  eval_metric:      ndcg@10
  learning_rate:    0.05
  n_estimators:     200
  max_depth:        6
  subsample:        0.9
  colsample_bytree: 0.9
  reg_lambda:       1.0
```

Multiple versioned configs allow structured hyperparameter sweeps tracked by MLflow.

---

#### 5.2.4 MLflow Integration

- **Tracking URI**: `http://127.0.0.1:5000` (local SQLite backend: `mlflow.db`)
- **Artifact Storage**: `/app/local/mlartifacts`
- **Logged Data**: All nested YAML params (flattened to `training.learning_rate`, etc.), NDCG@10 metric, model artifact, config file.
- **Run Comparison**: MLflow UI enables side-by-side comparison of all training runs.

---

### 5.3 AutoGym-Serving: Model Inference

**Repository**: `AutoGym-serving/sparky-serving/`

#### 5.3.1 Serving Backends

| Backend | File | Description | Use Case |
|---|---|---|---|
| Baseline | `app_baseline.py` | Direct XGBoost inference | Development, debugging |
| Optimized | `app_optimized.py` | LRU model cache + pre-allocated NumPy buffers (≤128 instances) | High-throughput CPU |
| ONNX | `app_onnx.py` | ONNX Runtime CPU session | Cross-platform, low-latency CPU |
| Triton | `triton/model_repository/` | NVIDIA Triton Inference Server | GPU inference, large-scale |

---

#### 5.3.2 Feature Assembly (Inference-Time)

Mirrors the training pipeline exactly to prevent train/serve skew:
1. Extract `user_id` and `recipe_id` from the instance dict.
2. For each of 44 feature columns: fill missing with `0.0`.
3. String columns: sorted-unique integer mapping (unknown → -1).
4. All values → `float32`.
5. Set XGBoost group size = number of instances.
6. Predict and sort by score descending → assign ranks 1..N.

---

#### 5.3.3 Optimizations in `app_optimized.py`

```python
@lru_cache(maxsize=1)
def get_model(model_path: str) -> xgb.Booster:
    ...  # Single model instance, cached permanently

# Pre-allocated buffers for small batches
if n_instances <= 128:
    buf = np.zeros((128, len(FEATURE_COLS)), dtype=np.float32)
    buf[:n_instances] = feature_matrix
    feature_matrix = buf[:n_instances].copy()
```

---

#### 5.3.4 Prometheus Monitoring

All FastAPI servers instrument:
- `http_requests_total` — by endpoint and status code
- `http_request_duration_seconds` — histogram with percentiles
- `http_request_size_bytes` — request payload size
- Exposed at `/metrics` for Prometheus scraping

---

#### 5.3.5 Benchmarking (`scripts/benchmark.py`)

- Generates synthetic `PredictRequest` payloads with variable batch sizes.
- Tests latency at multiple concurrency levels.
- Captures P50, P95, P99 latency percentiles.
- Compares across baseline / ONNX / optimized / Triton backends.
- Outputs results to `results/` as JSON + CSV.

---

## 6. Data Strategy

### 6.1 Source Data

| Dataset | Source | Size | Description |
|---|---|---|---|
| RAW_recipes.csv | Kaggle Food.com | ~230K rows | Recipes with nutrition, tags, ingredients |
| RAW_interactions.csv | Kaggle Food.com | ~1.1M rows | User-recipe ratings (1–5 stars) |

### 6.2 Temporal Split Strategy

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

### 6.3 Label Definition

- **Positive** (`label=1`): `rating ≥ 4` — user cooked and enjoyed the recipe.
- **Negative** (`label=0`): `rating < 4` — user did not find the recipe satisfying.
- Expected positive rate: ~40–60% across splits.

### 6.4 Versioning & Reproducibility

Every training run is tied to an immutable artifact:

```
manifest.json
  ├── pipeline_version: "1.0.0"
  ├── git_commit: "<40-char SHA>"
  ├── timestamp: "<ISO-8601>"
  └── splits:
        ├── train.sha256 → exact bytes of train.csv
        ├── val.sha256   → exact bytes of val.csv
        └── test.sha256  → exact bytes of test.csv
```

Given any manifest, the exact dataset used for training can be recovered from Swift object storage.

---

## 7. Machine Learning Design

### 7.1 Algorithm: XGBoost LambdaRank

**Objective**: `rank:ndcg`

LambdaRank is a pairwise learning-to-rank algorithm that directly optimizes NDCG by computing gradient updates using pairwise preference differences weighted by the change in NDCG from swapping the pair.

**Why LambdaRank for this problem**:
- The task is to rank recipes for a user, not to predict click probability.
- NDCG penalizes placing high-relevance items lower in the list more than displacing low-relevance items.
- Group-based training naturally models the per-user ranking context.

### 7.2 User History Embeddings

Raw cooking history is a sparse, high-dimensional signal (user × cuisine one-hot). PCA reduces this to 6 latent dimensions that capture the dominant cooking preference patterns:

```python
from sklearn.decomposition import PCA

# cuisine_matrix: (n_users, n_cuisines) one-hot encoded
pca = PCA(n_components=6)
history_embeddings = pca.fit_transform(cuisine_matrix)
# → (n_users, 6) dense embedding
```

This provides the model with a compact representation of "who this user is" as a cook, enabling generalization to unseen recipe-user combinations.

### 7.3 Model Variants

| Config | Description | Expected NDCG@10 |
|---|---|---|
| `baseline_popularity` | Rank by `n_positive_ratings` | Baseline (no ML) |
| `xgb_ranker_v1` | XGBoost, 200 trees, depth 6 | ~0.78 |
| `xgb_ranker_ray` | Distributed Ray training (4 workers) | Same as v1 |
| `xgb_ranker_v2+` | Tuned hyperparameters via MLflow sweeps | Iteratively higher |

### 7.4 Inference Logic

```
PredictRequest.instances (list of 44-feature dicts)
    ↓
assemble_features() → float32 matrix + user_ids + recipe_ids
    ↓
DMatrix(features, group=[len(instances)])
    ↓
booster.predict(dmatrix)  → raw scores
    ↓
sort by score descending → assign ranks 1..N
    ↓
PredictResponse.predictions [{user_id, recipe_id, score, rank}]
```

---

## 8. Infrastructure & DevOps

### 8.1 Compute Environment

| Resource | Platform | Purpose |
|---|---|---|
| KVM@TACC | Chameleon Cloud | ML training (CPU/GPU VMs) |
| CHI@TACC | Chameleon Cloud (OpenStack) | Swift object storage |
| Local Docker | Developer workstations | Testing and development |

### 8.2 Container Architecture

Each sub-project is fully containerized:

**sparky-data-pipeline** (`docker-compose.yml`):

| Service | Profile | Description |
|---|---|---|
| `postgres` | always | PostgreSQL 15, port 5433 |
| `ingest` | ingest | Raw data upload to Swift |
| `data-generator` | generate | Synthetic interaction replay |
| `online-feature-demo` | online-demo | Feature pipeline demo |
| `batch-pipeline` | batch | Versioned dataset compilation |
| `soda-checks` | soda | Data quality validation |

**AutoGym-main**:
- `Dockerfile` — Training container (Python 3.11, XGBoost, Ray)
- `Dockerfile.mlflow` — MLflow tracking server
- `docker-compose.mlops.yml` — Composes training + MLflow server

**AutoGym-serving**:
- `Dockerfile.baseline` / `.onnx` / `.optimized` / `.triton` — Four variants
- `docker-compose-baseline/onnx/optimized/triton.yaml` — Per-variant deployment

### 8.3 Orchestration (Makefile)

Full pipeline execution:
```bash
make all
# 1. make setup        → init postgres, create output/
# 2. make build        → docker compose build
# 3. make enrich       → run build_training_table.py
# 4. make ingest       → upload raw data to Swift
# 5. make generate     → replay interactions to PostgreSQL
# 6. make online-demo  → demo OnlineFeaturePipeline
# 7. make batch        → compile versioned training dataset
```

Model serving:
```bash
make build-optimized && make run-optimized
# Starts optimized FastAPI server on port 8000
# Starts Jupyter on port 8888
```

---

## 9. API Contracts

### 9.1 Health Check

**`GET /health`**
```json
{
  "status": "healthy",
  "model": "xgb_ranker",
  "backend": "xgboost_optimized"
}
```

### 9.2 Predict

**`POST /predict`**

**Request**:
```json
{
  "request_id": "req_20260409_001",
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
      "has_milk": 1,
      "has_egg": 0,
      "has_nuts": 0,
      "has_peanut": 0,
      "has_fish": 0,
      "has_shellfish": 0,
      "has_wheat": 1,
      "has_soy": 0,
      "has_sesame": 0,
      "daily_calorie_target": 2200.0,
      "protein_target_g": 60.0,
      "carbs_target_g": 275.0,
      "fat_target_g": 65.0,
      "user_vegetarian": 0,
      "user_vegan": 0,
      "user_gluten_free": 0,
      "user_dairy_free": 0,
      "user_low_sodium": 0,
      "user_low_fat": 0,
      "history_pc1": 0.42,
      "history_pc2": -0.11,
      "history_pc3": 0.07,
      "history_pc4": 0.22,
      "history_pc5": -0.03,
      "history_pc6": 0.15
    }
  ]
}
```

**Response**:
```json
{
  "request_id": "req_20260409_001",
  "model_name": "xgb_ranker",
  "model_version": "v1",
  "generated_at": "2026-04-09T14:30:00Z",
  "predictions": [
    {"user_id": 37449, "recipe_id": 33096, "score": 1.8421, "rank": 1},
    {"user_id": 37449, "recipe_id": 120964, "score": 0.9174, "rank": 2}
  ]
}
```

---

## 10. Evaluation & Metrics

### 10.1 Primary Metric: NDCG@10

Normalized Discounted Cumulative Gain at rank 10. Measures the quality of the top-10 ranked results for each user, penalizing relevant items placed lower in the list.

```
DCG@10  = Σ(rel_i / log2(i+1))  for i = 1..10
IDCG@10 = DCG@10 with ideal ordering
NDCG@10 = DCG@10 / IDCG@10

Final NDCG@10 = mean(NDCG@10 per user)
```

**Implementation** (`src/metric.py`):
- Groups predictions by `user_id`
- Sorts by predicted score descending
- Computes DCG/IDCG pair per user
- Returns macro-average across all users

### 10.2 Baseline Comparison

| Model | Description | NDCG@10 |
|---|---|---|
| Popularity Baseline | Rank by total positive ratings | Computed at run time |
| XGBoost Ranker v1 | Trained model | Computed at run time |
| XGBoost Ranker (tuned) | Best MLflow run | Iteratively improves |

### 10.3 Latency Targets

| Backend | P50 Target | P99 Target |
|---|---|---|
| Baseline | < 80 ms | < 150 ms |
| Optimized | < 50 ms | < 100 ms |
| ONNX | < 40 ms | < 80 ms |
| Triton (GPU) | < 20 ms | < 50 ms |

---

## 11. Scalability & Performance

### 11.1 Training Scalability (Ray)

- Horizontal scaling: shard users across N workers (modulo hash on `user_id`).
- Default config: 4 workers × 2 CPUs each.
- Fault tolerance: 1 worker failure tolerated; automatic checkpoint recovery every 25 rounds.
- Supports KVM@TACC cluster deployment.

### 11.2 Serving Scalability

- **Stateless API**: All serving backends are stateless; horizontal scaling via load balancer is trivial.
- **Model Loading**: LRU-cached via `@lru_cache(maxsize=1)` — zero per-request model I/O overhead.
- **Buffer Pooling**: Pre-allocated NumPy buffers for small batches (≤ 128 instances) avoid per-request GC pressure.
- **Triton Backend**: NVIDIA Triton supports dynamic batching, model ensemble, and concurrent model execution for GPU utilization.

### 11.3 Data Pipeline Scalability

- Synthetic data expansion: 2× by default; configurable `EXPAND_FACTOR`.
- Batch pipeline: Incremental merges — only new PostgreSQL interactions are appended per cycle.
- Online feature pipeline: User features cached in both memory (`dict`) and PostgreSQL for low-latency lookup.

---

## 12. MLOps & Continuous Learning

### 12.1 The Retraining Flywheel

```
1. Users interact with the recommendation system
       ↓
2. Actions logged to PostgreSQL (user_interactions)
       ↓
3. batch_pipeline.py (daily or triggered)
   → Merges new interactions with historical data
   → Runs Soda quality checks
   → Generates versioned manifest.json
       ↓
4. AutoGym training job consumes manifest
   → Reads exact dataset version (by hash)
   → Trains new XGBoost model
   → Logs to MLflow
       ↓
5. Best model deployed to AutoGym-Serving
       ↓
6. Updated predictions served to users
       ↓
   [Repeat from step 1]
```

### 12.2 Experiment Management (MLflow)

- Every training run logs: all YAML config parameters, NDCG@10, model artifact, config file, timestamp.
- MLflow UI provides side-by-side comparison across all runs.
- Model registry enables staging → production promotion workflow.

### 12.3 Data Quality Gates

Soda Core checks act as a mandatory gate:
- If any check fails (e.g., unexpected schema change from Kaggle, extreme label imbalance), the pipeline halts and alerts the team.
- Prevents corrupted or drifted data from silently degrading model quality.

### 12.4 Artifact Lineage

Full provenance chain for any trained model:
```
model artifact
    → MLflow run_id
    → manifest.json
    → train.csv SHA-256 + val.csv SHA-256 + test.csv SHA-256
    → batch_pipeline Git commit hash
    → PostgreSQL snapshot at manifest timestamp
```

---

## 13. Technology Stack Summary

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Data Processing** | Pandas | 2.2.0+ | ETL, feature engineering |
| | NumPy | 1.26.0+ | Numerical operations |
| | Scikit-learn | 1.3+ | PCA, preprocessing |
| **ML Training** | XGBoost | 2.1.4 | LambdaRank model |
| | Ray | 2.51.2 | Distributed training |
| **Experiment Tracking** | MLflow | 3.1.0 | Run tracking, model registry |
| **API Server** | FastAPI | 0.110.0+ | REST inference endpoints |
| | Uvicorn | 0.29.0+ | ASGI server |
| | Pydantic | 2.0+ | Request/response validation |
| **Model Runtimes** | ONNX Runtime | latest | CPU-optimized inference |
| | NVIDIA Triton | latest | GPU inference server |
| **Databases** | PostgreSQL | 15 | User interactions, feature cache |
| | SQLAlchemy / psycopg2 | — | ORM + direct queries |
| **Object Storage** | Chameleon OpenStack Swift | — | Versioned dataset storage |
| **Data Quality** | Soda Core | — | Automated validation checks |
| **Monitoring** | Prometheus | — | Metrics scraping |
| | FastAPI Instrumentator | 7.0.0+ | Auto-instrumentation |
| **Containerization** | Docker | — | Reproducible environments |
| | Docker Compose | — | Multi-service orchestration |
| **Compute** | Chameleon Cloud (KVM@TACC) | — | VM-based training/serving |
| **Languages** | Python | 3.11 | All components |
| **Config Format** | YAML | — | Training hyperparameters |

---

## 14. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Train/serve skew** — feature encoding diverges between training and inference | Medium | High | Shared encoding logic in both `ranking_data.py` and `app_*.py`; integration tests verify output matches |
| **Data leakage** — future data contaminates training | Medium | High | Time-stratified splits enforced in `build_training_table.py`; batch pipeline validates `train_max < test_min` |
| **Stale features** — user feature cache goes stale | Low | Medium | `updated_at` timestamp on `user_features_cache`; fallback to live PostgreSQL query if cache age > threshold |
| **Bad upstream data** — Kaggle schema changes silently | Medium | High | Soda Core schema + range checks block pipeline on violation |
| **Model degradation over time** — data drift | Medium | Medium | MLflow tracks NDCG@10 over retraining runs; alerts if metric drops below threshold |
| **Object storage unavailability** — Swift offline | Low | High | Local filesystem fallback in `ingest_to_object_store.py` |
| **Ray worker failure** | Low | Low | `max_failures=1` in distributed config; checkpoint recovery every 25 rounds |
| **Allergen safety violation** — user served allergen | Very Low | Critical | Rule filter applied before ML scoring; allergen flags are hard constraints, not soft features only |

---

## 15. Deliverables

### Completed

- [x] End-to-end data pipeline (`sparky-data-pipeline`) with ingestion, feature engineering, synthetic generation, batch compilation, and data quality checks
- [x] XGBoost LambdaRank training system (`AutoGym-main`) with MLflow tracking, Ray distributed training, and configurable hyperparameters
- [x] Production REST API (`AutoGym-serving`) with 4 serving backends (baseline, optimized, ONNX, Triton), Prometheus metrics, and benchmarking tooling
- [x] Temporal-correct train/val/test splits preventing data leakage
- [x] SHA-256 dataset versioning with full artifact lineage via `manifest.json`
- [x] PostgreSQL schema with optimized indexes for online feature lookups
- [x] Fully containerized stack with Docker Compose and Makefile orchestration
- [x] Allergen safety enforcement as hard rule filters before ML ranking
- [x] PCA-based user cooking history embeddings (6 components)
- [x] Continuous learning flywheel (data generator → batch pipeline → retraining)

### Future Work

#### Join Responsibilties
- [ ] The end-to-end plumbing needed for operation is in place, including movement of production data through the system, capture of feedback or outcomes, preparation of training data, retraining, evaluation, packaging, deployment, and rollback or update. These workflows should run through automation with minimal human work (e.g. you can choose to require manual approval of promotion from a canary environment to production, but not “engineer must SSH into the instance and run these commands to promote from canary environment to production”.)
- [ ] The complementary ML feature is implemented in the selected open source service itself, so that the feature is used in the regular user flow. (You may find it helpful to use an AI agent to help with this part, since the open source service may be a large, complex, codebase and possibly written in a language you are unfamiliar with.)
- [ ] “Bonus” items from the previous stage should also be integrated, in order for them to be credited in the “initial implementation” stage.
- [ ] Safeguarding plan: the team must deliver a safeguarding plan and implement it within the system. This plan should take active steps with concrete mechanisms to support fairness, explainability, transparency, privacy, accountability, and robustness principles (as discussed in lecture).

---


