# Model Card: SparkyFitness Meal Ranker

## Overview

The production model is an XGBoost LambdaRank recommender registered in MLflow
as `sparky-ranker`. It ranks candidate SparkyFitness meals for a user based on
nutrition targets, dietary flags, allergen indicators, recipe metadata, and
six PCA history features.

## Intended Use

- Recommend meals inside the SparkyFitness Foods page.
- Rank a candidate set prepared by the SparkyFitness backend.
- Support meal discovery and logging, not medical or clinical nutrition advice.

## Inputs and Outputs

- Input contract: `user_id`, `recipe_id`, and the 45 features in
  `src/serving/feature_contract.py`.
- Request metadata: `request_id` and `recommendation_id` are logged for
  observability and feedback joins, but are excluded from model features.
- Output: ranked candidates with `score`, `rank`, `model_version`, and
  `model_source`.

## Evaluation

Primary offline metric:

- NDCG@10 on held-out users/requests.

Quality gates before registration:

- NDCG@10 must meet the configured threshold.
- Model must not regress beyond the configured production tolerance.
- Fairness gate must pass group NDCG@10 checks.
- Allergen safety must pass the top-k restricted-user check.

## Safeguards

- Hard allergen flags are evaluated before model registration.
- SHAP explanations are generated after training and available at serving time.
- Serving logs model version and source for every prediction.
- Production can roll back to the previous archived MLflow Registry version.
- Grafana can call the authenticated rollback webhook for configured critical
  alerts when `ROLLBACK_WEBHOOK_TOKEN` is set.

## Monitoring

Prometheus and Grafana track:

- Prediction volume and latency.
- Error rate.
- Score distribution.
- Model load/reload timestamp.
- Prediction logging freshness.
- Retraining status and failures.
- PostgreSQL health.

## Limitations

- Recommendations depend on the SparkyFitness meal candidate pool; missing or
  sparse meal nutrition data can limit personalization.
- The model is optimized for ranking quality, not causal dietary impact.
- Feedback is behavioral and can be biased by UI placement and user traffic mix.

## Ownership

- Data: ingestion quality, training set compilation, live drift checks.
- Training: model quality, fairness gates, MLflow registration.
- Serving: online metrics, rollback/promotion triggers, feedback capture.
