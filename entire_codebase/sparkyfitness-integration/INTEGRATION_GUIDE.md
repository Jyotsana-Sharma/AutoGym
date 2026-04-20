# SparkyFitness ML Recommendation Integration Guide

This directory contains the patch set that adds the personalized recipe
recommendation feature to the SparkyFitness application. In the integrated
AutoGym deployment this patch is applied automatically by the
`sparkyfitness-setup` container before the SparkyFitness server and frontend
start.

---

## Files

```
sparkyfitness-integration/
├── apply_integration.py
├── SparkyFitnessServer/
│   ├── schemas/recommendationSchemas.ts
│   ├── models/recommendationRepository.ts
│   ├── services/recommendationService.ts
│   ├── routes/recommendationRoutes.ts
│   └── db/add_recommendations.sql
└── SparkyFitnessFrontend/
    └── src/
        ├── api/recommendations.ts
        ├── hooks/Foods/useRecommendations.ts
        └── pages/Foods/RecipeRecommendations.tsx
```

---

## Automatic Integration

The preferred path is to let Docker Compose apply the integration:

```bash
cd entire_codebase
docker compose --profile pipeline up -d
```

The `sparkyfitness-setup` service runs:

```bash
python3 /integration/apply_integration.py
```

It is idempotent and performs four steps:

1. Copies backend route, service, repository, and schema files into `SparkyFitnessServer/`.
2. Copies frontend API, hook, and `RecipeRecommendations` component into `SparkyFitnessFrontend/`.
3. Patches `SparkyFitnessServer.ts` with:

```ts
import recommendationRoutes from './routes/recommendationRoutes.js';
app.use('/api/recommendations', recommendationRoutes);
```

4. Patches `Foods.tsx` to render:

```tsx
<RecipeRecommendations limit={6} />
```

---

## Manual Integration

Use this path only if you are applying the patch to a separately cloned
SparkyFitness repository.

```bash
INTEG=/path/to/AutoGym/entire_codebase/sparkyfitness-integration
SF=/path/to/SparkyFitness

APP_DIR=$SF \
INTEGRATION_DIR=$INTEG \
ENV_EXAMPLE=/path/to/AutoGym/entire_codebase/.env.sparky.example \
python3 $INTEG/apply_integration.py
```

Verify:

```bash
grep -n "recommendationRoutes" \
  $SF/SparkyFitnessServer/SparkyFitnessServer.ts

grep -n "RecipeRecommendations" \
  $SF/SparkyFitnessFrontend/src/pages/Foods/Foods.tsx
```

---

## Database Tables

The application repository creates recommendation tables on demand, and the
SQL migration is available if you want to apply it explicitly:

```bash
psql -U sparky -d sparkyfitness_db \
  -f sparkyfitness-integration/SparkyFitnessServer/db/add_recommendations.sql
```

Tables:

| Table | Purpose |
|---|---|
| `recommendation_cache` | Stores the ranked meals returned to a user, with model version and score |
| `recommendation_interactions` | Stores user actions: `viewed`, `logged`, `dismissed`, `saved` |

The migration enables `pgcrypto` so `gen_random_uuid()` works on a fresh
PostgreSQL instance.

---

## Environment

The SparkyFitness server needs the ML serving endpoint:

```bash
ML_RECOMMENDATION_URL=http://sparky-serving:8000
ML_MODEL_NAME=sparky-ranker
```

In the unified Compose deployment, `sparkyfitness-server` is attached to both
the SparkyFitness network and `sparky-net`, so it can reach the ML service by
the internal name `http://sparky-serving:8000`.

---

## End-To-End Flow

```
User opens Foods page
      │
      ▼
RecipeRecommendations (React)
  GET /api/recommendations?limit=12
      │
      ▼
recommendationRoutes (Express)
  → recommendationService
      ├─ getUserGoals()
      ├─ getRecentlyLoggedMealIds()
      └─ getCandidateMeals()
      │
      ▼
Stable numeric surrogate IDs generated for ML request
      │
      ▼  POST /predict
SparkyFitness ML serving API (FastAPI, port 8000)
      └─ XGBoost LambdaRank scores each meal
      │
      ▼
Predicted numeric IDs are mapped back to SparkyFitness meal UUIDs
      │
      ▼
Top-N ranked meals returned to frontend
      │
      ▼
User clicks Add / Save / Dismiss
      │
      ▼
POST /api/recommendations/feedback
      │
      ▼
recommendation_interactions
```

The ML serving API also logs prediction outputs and raw feature values to the
ML PostgreSQL database. Those logs feed drift monitoring and retraining.
App-side UI feedback is stored in the SparkyFitness database and forwarded to
the ML `/feedback` endpoint. The shared `request_id` and `recommendation_id`
fields allow SparkyFitness `recommendation_interactions` to join with ML-side
`prediction_log`, `user_feedback`, and `inference_features`.

---

## Graceful Degradation

If the ML service is unreachable, `recommendationService.ts` falls back to a
protein-proximity heuristic. The Foods page still renders recommendations, and
the response reports `model_version: "fallback"`.

---

## Optional Future Enhancements

| Improvement | Where |
|---|---|
| Explicit dietary/allergen preferences | `buildFeatureVector()` and SparkyFitness profile tables |
| PCA history embeddings from live app data | `recommendationService.ts` before `/predict` |
| Meal-type slot filtering | Query params + candidate SQL filtering |
