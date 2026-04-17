# SparkyFitness ML System — Safeguarding Plan

## Overview

SparkyFitness is a personalized recipe recommendation system that uses an
XGBoost learning-to-rank model to suggest meals to users based on dietary
preferences, allergen restrictions, nutritional targets, and cooking history.

This plan describes concrete, implemented mechanisms to uphold six safeguarding
principles: **fairness**, **explainability**, **transparency**, **privacy**,
**accountability**, and **robustness**.

---

## 1. Fairness

### Risk
Users with dietary restrictions (vegan, gluten-free, allergen sensitivities)
might receive worse recommendations than unrestricted users if the model
under-learns their preferences due to smaller representation in training data.

### Mechanisms Implemented

**Post-training fairness gate** (`safeguarding/fairness_checker.py`)
- After every training run, NDCG@10 is computed per dietary group
  (vegetarian, vegan, gluten-free, dairy-free, low-sodium, low-fat)
- Any group whose NDCG@10 drops > 20% below the overall must fail the quality gate
- Models that fail this fairness gate are **not registered** to the MLflow Registry
- Groups with < 50 users are excluded (insufficient statistical power)

**Allergen safety check**
- Verified that top-5 recommendations for allergen-restricted users do not
  contain their allergens at a rate > 1%
- Treated as a hard safety gate (not a soft warning)

**Bias-aware label construction** (`sparky-data-pipeline/build_training_table.py`)
- Labels derived from ratings (≥ 4 → positive), not from clicks alone
- User-specific macro targets ensure personalization is baked into features
- PCA history embeddings prevent any single cuisine from dominating

---

## 2. Explainability

### Risk
Users may not understand why a recipe was recommended; engineers cannot
debug unexpected recommendations.

### Mechanisms Implemented

**Per-prediction explanations** (`safeguarding/explainability.py`)
- SHAP TreeExplainer generates per-instance feature attributions
- Serving API's `/explain` endpoint returns top-10 contributing features
- User-facing explanation text auto-generated: "This recipe was recommended
  because your protein target, cooking history, and cuisine preference all
  matched"
- Rule-based fallback when SHAP is unavailable (latency-sensitive paths)

**Global feature importance** (logged to MLflow at every training run)
- Mean |SHAP| across test set logged as artifact `explainability/global_feature_importance.json`
- Feature importance plot (`shap_summary.png`) archived with the model run
- Enables auditors to verify that allergen flags are used correctly
  (e.g., `has_milk` should have positive importance for dairy-tolerant users
  and near-zero or negative importance for dairy-free users)

---

## 3. Transparency

### Risk
Stakeholders (users, regulators, team members) cannot see what model version
is running, how it was trained, or what data it used.

### Mechanisms Implemented

**Model version in every response**
- Every `/predict` response includes `model_version` and `model_source`
  fields showing exactly which registered MLflow version is serving predictions

**MLflow experiment tracking** (every training run)
- All hyperparameters logged
- Dataset hash (SHA-256) from `manifest.json` logged, ensuring reproducibility
- Config YAML logged as artifact
- Quality gate results logged as `quality_gate_results.json`
- Fairness check results logged as `fairness_results.json`

**Data versioning** (`sparky-data-pipeline/batch_pipeline.py`)
- Every training dataset has a manifest with SHA-256 hashes and Git commit
- Training data never overwritten in-place; each version has a unique ID
- Object storage (Chameleon Swift) retains all historical versions

**Public model lineage**
- MLflow Registry shows: training run → data version → model version → serving deployment
- Full audit trail queryable via MLflow UI (port 5000)

---

## 4. Privacy

### Risk
User dietary restrictions, allergen data, and nutritional goals are
health-adjacent personal information that must be handled carefully.

### Mechanisms Implemented

**No PII in model artifacts**
- Trained model (XGBoost JSON) contains no user data — only learned weights
- Feature vectors contain aggregated/anonymized user signals (PCA embeddings,
  averaged macro targets), not raw interaction logs

**Pseudonymized user IDs**
- All pipeline tables use integer `user_id` as the only user identifier
- No email, name, or device information stored in any ML pipeline table

**Prediction log retention policy** (`prediction_log` table)
- `inference_features` table auto-deleted after 90 days (implement as a
  PostgreSQL cron job: `DELETE FROM inference_features WHERE captured_at < NOW() - INTERVAL '90 days'`)
- Drift monitoring uses aggregate statistics, not individual row lookups

**Minimal feature collection**
- Only the 44 features needed for the ranking task are logged at inference
- No request metadata (IP, user-agent, session) is logged to ML tables

**Schema isolation**
- ML tables (`prediction_log`, `user_feedback`, `inference_features`) are
  logically separate from any application user table that may contain PII

---

## 5. Accountability

### Risk
When the model behaves badly (wrong recommendations, allergen violations),
it must be possible to determine who approved the deployment and trace it back.

### Mechanisms Implemented

**Quality gate approval trail**
- Every model in the MLflow Registry has `quality_gate_status` and `registered_at` tags
- Promotion from Staging → Production is either:
  - Manual: triggered via GitHub Actions `production` environment (requires approval)
  - Automatic (weekly schedule only): logged in `retraining_log` table

**Retraining log** (`retraining_log` table)
- Every retraining run records: trigger reason, run_id, model version,
  NDCG@10, registered, promoted, duration, status

**Rollback audit trail**
- `model_registry.py rollback()` logs the replacement in MLflow model description
- Previous production version archived (not deleted) — always available for comparison

**Alert accountability**
- `alertmanager.yml` routes alerts to webhook; receiver logs alert to `retraining_log`
- All automated actions (rollback, retraining) are logged with timestamps

---

## 6. Robustness

### Risk
Model performance can degrade due to data drift, infrastructure failures,
or adversarial inputs.

### Mechanisms Implemented

**Data drift monitoring** (`sparky-data-pipeline/scripts/drift_monitor.py`)
- Kolmogorov-Smirnov test on 19 numeric features every 5 minutes
- Drift detected if > 30% of features show statistically significant shift (p < 0.05)
- Automatic retraining trigger sent to retrain-api on detection

**Automated rollback** (Prometheus alert + serving API)
- Alert: `LowPredictionScores` fires if median score < 0.2 for 15+ minutes
- Alert: `HighErrorRate` fires if >5% of requests fail for 5+ minutes
- On alert: Alertmanager calls `/admin/rollback` via webhook → model rolls back
  to previous Production version without human intervention

**Model fallback**
- If MLflow Registry is unreachable, serving falls back to a local model file
  (`MODEL_FALLBACK_PATH`) — no 503s during MLflow downtime

**Quality gates before registration**
- NDCG@10 ≥ 0.55 (absolute threshold)
- NDCG@10 must not regress > 1% vs. current Production
- Fairness gate (see §1)
- Soda data quality gate (see data pipeline)

**Input validation**
- Pydantic validates every prediction request schema
- Missing feature values default to 0.0 (safe neutral value)
- Empty instance lists rejected with 400

**Infrastructure health monitoring**
- Prometheus + Grafana monitor serving latency (p95), error rate, DB connections,
  disk usage, memory, and MLflow availability
- Alertmanager with escalating severity levels (info → warning → critical)

---

## Summary Table

| Principle       | Mechanism | Location | Automated? |
|-----------------|-----------|----------|-----------|
| Fairness        | Group NDCG@10 gate, allergen safety | `safeguarding/fairness_checker.py` | Yes (post-training) |
| Explainability  | SHAP per-request + global importance | `safeguarding/explainability.py` | Yes |
| Transparency    | Model version in response, MLflow lineage | `app_production.py`, MLflow | Yes |
| Privacy         | Pseudonymized IDs, no PII in models, 90-day retention | DB schema, drift monitor cleanup | Yes |
| Accountability  | Retraining log, approval gates, rollback trail | `retraining_log`, GitHub Environments | Yes |
| Robustness      | Drift monitoring, auto-rollback, fallback model | `drift_monitor.py`, alerts, `model_loader.py` | Yes |

---

## Known Gaps and Next Steps

1. **Differential privacy**: For larger user bases, consider DP-SGD or RAPPOR for user feature aggregation.
2. **Counterfactual explanations**: Supplement SHAP with "if you liked more Italian food, these recipes would rank higher" style explanations.
3. **Adversarial robustness**: Add input perturbation testing to catch unusually large feature values that could manipulate rankings.
4. **External audit**: Document model card and submit to team lead review before production launch.
