# SparkyFitness ML Recommendation Integration Guide

This directory contains all files needed to add the **personalised recipe
recommendation** feature to the SparkyFitness application.

Important clarification:
- This directory is an **integration patch set**, not a standalone copy of the full upstream SparkyFitness repo.
- The unified ML system lives in `entire_codebase/`.
- These files show exactly how the open-source service calls the ML system in the regular user flow.

---

## Directory map → where each file goes

```
sparkyfitness-integration/
├── SparkyFitnessServer/
│   ├── schemas/recommendationSchemas.ts   → SparkyFitnessServer/schemas/
│   ├── models/recommendationRepository.ts → SparkyFitnessServer/models/
│   ├── services/recommendationService.ts  → SparkyFitnessServer/services/
│   ├── routes/recommendationRoutes.ts     → SparkyFitnessServer/routes/
│   └── db/add_recommendations.sql         → run once against your PostgreSQL DB
└── SparkyFitnessFrontend/
    └── src/
        ├── api/recommendations.ts         → SparkyFitnessFrontend/src/api/
        └── pages/Foods/
            └── RecipeRecommendations.tsx  → SparkyFitnessFrontend/src/pages/Foods/
```

---

## Step 1 — Run the database migration

```bash
psql -U sparky -d sparkydb -f SparkyFitnessServer/db/add_recommendations.sql
```

This creates two tables:
- `recommendation_cache` — stores ML-generated scores per user
- `recommendation_interactions` — feedback loop (viewed / logged / dismissed / saved)

`recommendation_interactions` captures application-level feedback inside SparkyFitness.
The ML system itself separately captures serving-level feedback through its `/feedback`
endpoint and stores it in `user_feedback` / `user_interactions`.

---

## Step 2 — Copy the new backend files

```bash
cp SparkyFitnessServer/schemas/recommendationSchemas.ts  <repo>/SparkyFitnessServer/schemas/
cp SparkyFitnessServer/models/recommendationRepository.ts <repo>/SparkyFitnessServer/models/
cp SparkyFitnessServer/services/recommendationService.ts  <repo>/SparkyFitnessServer/services/
cp SparkyFitnessServer/routes/recommendationRoutes.ts     <repo>/SparkyFitnessServer/routes/
```

---

## Step 3 — Mount the route in SparkyFitnessServer.ts

Find the section in `SparkyFitnessServer/SparkyFitnessServer.ts` where other
routers are imported and mounted (e.g. near `mealRoutes`, `foodRoutes`).
Add the two lines marked with `// ← ADD`:

```typescript
// ← ADD: import
import { createRecommendationRouter } from "./routes/recommendationRoutes";

// ← ADD: mount (pass the same pool/db object used by other routes)
app.use("/api", createRecommendationRouter(pool));
```

> **Auth middleware**: The route reads the authenticated user from
> `req.user?.id`.  SparkyFitness uses better-auth — verify that
> `req.user.id` is the correct property and adjust the two lines in
> `recommendationRoutes.ts` if needed.

---

## Step 4 — Add the environment variable

In `.env` (and `.env.example`):

```
# ML Recommendation service URL (your SparkyFitness ML Docker service)
ML_RECOMMENDATION_URL=http://localhost:8000
ML_MODEL_NAME=sparky-ranker
```

In `docker-compose.yml` for the `sparkyfitness-server` service add:

```yaml
environment:
  ML_RECOMMENDATION_URL: http://sparky-serving:8000
  ML_MODEL_NAME: sparky-ranker
```

> `sparky-serving` is the container name defined in the ML system's
> `docker-compose.yml`.  Both compose files must share a Docker network:
>
> ```yaml
> # in the SparkyFitness docker-compose.yml
> networks:
>   sparky-net:
>     external: true     # declared in the ML docker-compose.yml
> ```
>
> Or run both with a single compose using `include:` / `extends:`.

---

## Step 5 — Copy the frontend files

```bash
cp SparkyFitnessFrontend/src/api/recommendations.ts \
   <repo>/SparkyFitnessFrontend/src/api/

cp SparkyFitnessFrontend/src/pages/Foods/RecipeRecommendations.tsx \
   <repo>/SparkyFitnessFrontend/src/pages/Foods/
```

---

## Step 6 — Embed the component in the Foods page

Open `SparkyFitnessFrontend/src/pages/Foods/Foods.tsx` (or
`MealManagement.tsx`) and embed the panel where you want recommendations to
appear — for example, after the search bar:

```tsx
// ← ADD import
import RecipeRecommendations from "./RecipeRecommendations";

// ← ADD inside the JSX, e.g. below the meal search section
<RecipeRecommendations
  limit={6}
  onLogMeal={(mealId) => {
    // call your existing log-to-diary function here
    // e.g. handleAddMealToDiary(mealId)
  }}
/>
```

The component is self-contained: it fetches data, renders loading skeletons,
handles errors with a retry button, and logs user interactions automatically.

---

## How the feature works end-to-end

```
User opens Foods page
      │
      ▼
RecipeRecommendations (React)
  GET /api/recommendations?limit=12
      │
      ▼
recommendationRoutes (Express)
  → RecommendationService
      ├─ getUserGoals()            ← user_goals table
      ├─ getRecentlyLoggedMealIds() ← food_entries table
      └─ getCandidateMeals()       ← meals + meal_foods + food_variants
      │
      ▼  POST /predict
  SparkyFitness ML serving API (FastAPI, port 8000)
      └─ XGBoost LambdaRank scores each meal
      │
      ▼
  Top-N ranked meals returned to frontend
      │
      ▼
User clicks "Add to Diary" → feedback logged → recommendation_interactions
```

For milestone demos, show both parts:
- the **open-source service path** above, where the recommendation feature is used in the Foods page
- the **ML-system path** in `entire_codebase/`, where `/predict`, `/feedback`, retraining, evaluation, and monitoring run

---

## Graceful degradation

If the ML service is unreachable, `recommendationService.ts` automatically
falls back to a protein-proximity heuristic (meals closest to 30 % of the
user's protein target come first).  The frontend is unaffected — it just
receives a list of meals with `score: 0` and `model_version: "fallback"`.

---

## Future improvements

| Improvement | Where |
|---|---|
| Explicit dietary/allergen preferences | Add `user_dietary_preferences` table; wire into `inferDietaryFlags()` |
| PCA history embeddings | Compute from `food_entries` in `recommendationService.ts`; send to ML API |
| Allergen detection from ingredients | Parse ingredient names in `buildFeatureVector()` using `ALLERGEN_KEYWORDS` from `build_training_table.py` |
| Meal-type slot filtering | Pass `meal_type` through to ML API; retrain with meal-type feature |
| A/B testing | Use `model_version` field; route % of traffic to shadow model |
