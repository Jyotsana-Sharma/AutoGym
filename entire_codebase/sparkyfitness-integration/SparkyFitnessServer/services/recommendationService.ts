import { log } from '../config/logging.js';
import recommendationRepository, { MealCandidate, UserGoals, UserHistory } from '../models/recommendationRepository.js';
import { createHash, randomUUID } from 'crypto';

const ML_URL = process.env.ML_RECOMMENDATION_URL ?? 'http://localhost:8000';
const ML_MODEL_NAME = process.env.ML_MODEL_NAME ?? 'sparky-ranker';
const CANDIDATE_POOL_SIZE = 200;
// Minimum logged meals before we trust user history over goal defaults.
const MIN_HISTORY_ITEMS = 3;

// ---------------------------------------------------------------------------
// Feature engineering helpers
// ---------------------------------------------------------------------------
const PDV_REF = {
  calories: 2000,
  fat: 78,
  sugar: 50,
  sodium: 2.3,      // grams (2300 mg)
  protein: 50,
  saturated_fat: 20,
  carbohydrate: 275,
};

function toPDV(value: number | null, ref: number): number {
  if (value == null || ref === 0) return 0;
  return (value / ref) * 100;
}

interface UserHistoryFeatures {
  daily_calorie_target: number;
  protein_target_g: number;
  carbs_target_g: number;
  fat_target_g: number;
  user_vegetarian: number;
  user_vegan: number;
  user_gluten_free: number;
  user_dairy_free: number;
  user_low_sodium: number;
  user_low_fat: number;
  history_pc1: number;
  history_pc2: number;
  history_pc3: number;
  history_pc4: number;
  history_pc5: number;
  history_pc6: number;
}

// Derives the 16 user-level model features from logged food history.
// When history is thin (< MIN_HISTORY_ITEMS), falls back to goal values so
// the model still gets a meaningful macro signal.
function computeUserHistoryFeatures(history: UserHistory, goals: UserGoals): UserHistoryFeatures {
  const hasSufficientHistory = history.item_count >= MIN_HISTORY_ITEMS;

  const calTarget = hasSufficientHistory && history.avg_calories > 0
    ? history.avg_calories
    : goals.calories;
  const protTarget = hasSufficientHistory && history.avg_protein > 0
    ? history.avg_protein
    : goals.protein;
  const carbTarget = hasSufficientHistory && history.avg_carbs > 0
    ? history.avg_carbs
    : goals.carbs;
  const fatTarget = hasSufficientHistory && history.avg_fat > 0
    ? history.avg_fat
    : goals.fat;

  // Dietary flags derived from macro ratios in logged history.
  // Without ingredient-level tags we can only infer fat/sodium tendencies.
  const avgSodium = history.avg_sodium;
  const avgFat = history.avg_fat;
  const avgCal = history.avg_calories;
  const fatCalPct = avgCal > 0 ? (avgFat * 9) / avgCal : 0.35;

  const userLowSodium = hasSufficientHistory && avgSodium > 0 && avgSodium < 600 ? 1 : 0;
  const userLowFat = hasSufficientHistory && fatCalPct > 0 && fatCalPct < 0.30 ? 1 : 0;

  // Proxy history embeddings from macro proportions.
  // These aren't true PCA components but encode the same eating-pattern
  // dimensions the model was trained to read: protein density, carb density,
  // fat density, sodium tendency, fat tendency, and engagement level.
  const totalMacroKcal = (protTarget * 4) + (carbTarget * 4) + (fatTarget * 9) || 1;
  const pc1 = (protTarget * 4) / totalMacroKcal;   // protein share of calories
  const pc2 = (carbTarget * 4) / totalMacroKcal;   // carb share
  const pc3 = (fatTarget * 9) / totalMacroKcal;    // fat share
  const pc4 = userLowFat;
  const pc5 = userLowSodium;
  const pc6 = Math.min(history.item_count / 20, 1.0); // engagement depth (0–1)

  return {
    daily_calorie_target: calTarget,
    protein_target_g: protTarget,
    carbs_target_g: carbTarget,
    fat_target_g: fatTarget,
    user_vegetarian: 0,   // requires ingredient-level tags not available in this DB
    user_vegan: 0,
    user_gluten_free: 0,
    user_dairy_free: 0,
    user_low_sodium: userLowSodium,
    user_low_fat: userLowFat,
    history_pc1: pc1,
    history_pc2: pc2,
    history_pc3: pc3,
    history_pc4: pc4,
    history_pc5: pc5,
    history_pc6: pc6,
  };
}

function buildFeatureVector(
  meal: MealCandidate,
  userFeatures: UserHistoryFeatures,
): Record<string, number | string> {
  const cal = meal.calories ?? 0;
  const fat = meal.fat ?? 0;
  const sugar = meal.sugars ?? 0;
  const sodium_g = (meal.sodium ?? 0) / 1000; // mg → g
  const protein = meal.protein ?? 0;
  const sat_fat = meal.saturated_fat ?? 0;
  const carb = meal.carbs ?? 0;

  return {
    minutes: 30,
    n_ingredients: meal.n_ingredients,
    n_steps: 5,
    avg_rating: 0,
    n_reviews: 0,
    cuisine: 'unknown',
    calories: toPDV(cal, PDV_REF.calories),
    total_fat: toPDV(fat, PDV_REF.fat),
    sugar: toPDV(sugar, PDV_REF.sugar),
    sodium: toPDV(sodium_g, PDV_REF.sodium),
    protein: toPDV(protein, PDV_REF.protein),
    saturated_fat: toPDV(sat_fat, PDV_REF.saturated_fat),
    carbohydrate: toPDV(carb, PDV_REF.carbohydrate),
    total_fat_g: fat,
    sugar_g: sugar,
    sodium_g: sodium_g,
    protein_g: protein,
    saturated_fat_g: sat_fat,
    carbohydrate_g: carb,
    has_egg: 0, has_fish: 0, has_milk: 0, has_nuts: 0, has_peanut: 0,
    has_sesame: 0, has_shellfish: 0, has_soy: 0, has_wheat: 0,
    // User features — populated from real history instead of hardcoded zeros
    daily_calorie_target: userFeatures.daily_calorie_target,
    protein_target_g: userFeatures.protein_target_g,
    carbs_target_g: userFeatures.carbs_target_g,
    fat_target_g: userFeatures.fat_target_g,
    user_vegetarian: userFeatures.user_vegetarian,
    user_vegan: userFeatures.user_vegan,
    user_gluten_free: userFeatures.user_gluten_free,
    user_dairy_free: userFeatures.user_dairy_free,
    user_low_sodium: userFeatures.user_low_sodium,
    user_low_fat: userFeatures.user_low_fat,
    history_pc1: userFeatures.history_pc1,
    history_pc2: userFeatures.history_pc2,
    history_pc3: userFeatures.history_pc3,
    history_pc4: userFeatures.history_pc4,
    history_pc5: userFeatures.history_pc5,
    history_pc6: userFeatures.history_pc6,
  };
}

// ---------------------------------------------------------------------------
// Explainability
// ---------------------------------------------------------------------------

// Returns a human-readable sentence explaining the top reason this meal was
// recommended, based on the actual feature values sent to the model.
function reasonForRecommendation(meal: MealCandidate, userFeatures: UserHistoryFeatures): string {
  const protein = meal.protein ?? 0;
  const cal = meal.calories ?? 0;
  const fat = meal.fat ?? 0;
  const sodium_mg = meal.sodium ?? 0;
  const carb = meal.carbs ?? 0;

  const protTarget = userFeatures.protein_target_g;
  const calTarget = userFeatures.daily_calorie_target;
  const fatTarget = userFeatures.fat_target_g;
  const carbTarget = userFeatures.carbs_target_g;

  // Score each dimension by how closely it matches the user's targets.
  // Lower relative deviation = stronger reason to surface that dimension.
  const calPct = calTarget > 0 ? Math.round((cal / calTarget) * 100) : 0;

  if (protein >= 20 && protTarget > 0 && Math.abs(protein - protTarget * 0.3) / (protTarget * 0.3) < 0.25) {
    return `Matches your protein goal — ${protein.toFixed(0)} g (target: ${protTarget.toFixed(0)} g/day)`;
  }
  if (userFeatures.user_low_fat === 1 && fat > 0 && fat < fatTarget * 0.25) {
    return `Low fat — only ${fat.toFixed(0)} g, fits your low-fat eating pattern`;
  }
  if (userFeatures.user_low_sodium === 1 && sodium_mg > 0 && sodium_mg < 500) {
    return `Low sodium — ${sodium_mg.toFixed(0)} mg, fits your low-sodium eating pattern`;
  }
  if (calTarget > 0 && calPct >= 20 && calPct <= 35) {
    return `Balanced meal — ~${calPct}% of your ${calTarget.toFixed(0)} kcal daily goal`;
  }
  if (cal > 0 && cal <= calTarget * 0.15) {
    return `Light option — only ${cal.toFixed(0)} kcal`;
  }
  if (carb > 0 && carbTarget > 0 && Math.abs(carb - carbTarget * 0.3) / (carbTarget * 0.3) < 0.2) {
    return `Balanced carbs — ${carb.toFixed(0)} g (~30% of your ${carbTarget.toFixed(0)} g/day target)`;
  }
  if (protein >= 15) {
    return `Good protein source — ${protein.toFixed(0)} g per serving`;
  }
  if (calPct > 0 && calPct <= 20) {
    return `Light snack — ${cal.toFixed(0)} kcal`;
  }
  return 'Nutritionally aligned with your profile';
}

// ---------------------------------------------------------------------------
// Recommendation bucketing
// ---------------------------------------------------------------------------

// Assigns each ranked result to a named bucket:
//   "personalized"  — top ML picks strongly aligned to user history
//   "discover"      — solid ML score but macro profile differs from user's avg
//   "goals"         — best per-macro alignment to user targets (from full pool)
function assignBuckets(
  ranked: Array<{ meal: MealCandidate; score: number }>,
  userFeatures: UserHistoryFeatures,
  limit: number,
): Array<{ meal: MealCandidate; score: number; bucket: string }> {
  const calT = userFeatures.daily_calorie_target || 2000;
  const proT = userFeatures.protein_target_g || 50;
  const carbT = userFeatures.carbs_target_g || 250;
  const fatT = userFeatures.fat_target_g || 65;

  // Macro proximity score — lower is better (dimensionless relative deviation).
  function macroProximity(m: MealCandidate): number {
    const perMeal = 0.3; // assume a meal is ~30% of daily targets
    const cDev = calT > 0 ? Math.abs((m.calories ?? 0) - calT * perMeal) / (calT * perMeal) : 1;
    const pDev = proT > 0 ? Math.abs((m.protein ?? 0) - proT * perMeal) / (proT * perMeal) : 1;
    const carbDev = carbT > 0 ? Math.abs((m.carbs ?? 0) - carbT * perMeal) / (carbT * perMeal) : 1;
    const fDev = fatT > 0 ? Math.abs((m.fat ?? 0) - fatT * perMeal) / (fatT * perMeal) : 1;
    // Weight protein 2× — it's the most goal-sensitive macro
    return (cDev + pDev * 2 + carbDev + fDev) / 5;
  }

  const splitIdx = Math.ceil(limit * 0.6);

  // Top 60% by ML score → "personalized"
  // Bottom 40% → "discover" (good ML score, broader exploration)
  const withBuckets = ranked.map((r, i) => ({
    ...r,
    bucket: i < splitIdx ? 'personalized' : 'discover',
    _proximity: macroProximity(r.meal),
  }));

  // Re-label the item with best macro proximity as "goals" (even if it's
  // already in personalized — the client can choose how to display it).
  const bestGoalsIdx = withBuckets.reduce(
    (best, r, i) => (r._proximity < withBuckets[best]._proximity ? i : best),
    0,
  );
  withBuckets[bestGoalsIdx] = { ...withBuckets[bestGoalsIdx], bucket: 'goals' };

  return withBuckets.map(({ _proximity, ...rest }) => rest);
}

// ---------------------------------------------------------------------------
// ML API client
// ---------------------------------------------------------------------------
interface MLInstance extends Record<string, number | string> {
  user_id: number;
  recipe_id: number;
  recommendation_id: string;
}

interface MLResponse {
  predictions: Array<{ recipe_id: number | string; score: number }>;
  model_version?: string;
}

async function sendMLFeedback(payload: {
  request_id: string;
  recommendation_id: string;
  user_id: number;
  recipe_id: number;
  action: string;
}): Promise<void> {
  const response = await fetch(`${ML_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(5_000),
  });
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`ML feedback ${response.status}: ${text.slice(0, 200)}`);
  }
}

function stablePositiveInt(value: unknown): number {
  const digest = createHash('sha256').update(String(value)).digest();
  return digest.readUInt32BE(0) & 0x7fffffff;
}

async function callMLPredict(requestId: string, instances: MLInstance[]): Promise<MLResponse> {
  const response = await fetch(`${ML_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, model_name: ML_MODEL_NAME, instances }),
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`ML service ${response.status}: ${text.slice(0, 200)}`);
  }
  return response.json() as Promise<MLResponse>;
}

// ---------------------------------------------------------------------------
// Fallback ranking (no ML available, or cold start)
// ---------------------------------------------------------------------------
function rankByMacroProximity(
  candidates: MealCandidate[],
  userFeatures: UserHistoryFeatures,
  limit: number,
): Array<{ meal: MealCandidate; score: number }> {
  const calT = userFeatures.daily_calorie_target || 2000;
  const proT = userFeatures.protein_target_g || 50;
  const perMeal = 0.3;

  return [...candidates]
    .map(m => {
      const calDev = calT > 0 ? Math.abs((m.calories ?? 0) - calT * perMeal) / (calT * perMeal) : 1;
      const proDev = proT > 0 ? Math.abs((m.protein ?? 0) - proT * perMeal) / (proT * perMeal) : 1;
      const score = Math.max(0, 1 - (calDev + proDev * 2) / 3);
      return { meal: m, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

// ---------------------------------------------------------------------------
// Exported service functions
// ---------------------------------------------------------------------------
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getRecommendations(userId: any, limit: number, excludeRecentDays: number) {
  const [goals, recentlyLogged, history] = await Promise.all([
    recommendationRepository.getUserGoals(userId).catch(() => ({ calories: 2000, protein: 50, carbs: 250, fat: 65 })),
    recommendationRepository.getRecentlyLoggedMealIds(userId, excludeRecentDays).catch(() => new Set<string>()),
    recommendationRepository.getUserHistory(userId, 30).catch(() => ({
      item_count: 0, avg_calories: 0, avg_protein: 0, avg_carbs: 0, avg_fat: 0, avg_sodium: 0,
    })),
  ]);

  const userFeatures = computeUserHistoryFeatures(history, goals);

  let candidates = await recommendationRepository.getCandidateMeals(userId, recentlyLogged, CANDIDATE_POOL_SIZE);

  // Cold start: new user with no logged meals — serve globally popular public meals.
  const isColdStart = candidates.length === 0 || history.item_count < MIN_HISTORY_ITEMS;
  if (candidates.length === 0) {
    log('info', `[RecommendationService] Cold start for user ${userId} — fetching popular public meals`);
    candidates = await recommendationRepository.getPopularPublicMeals(CANDIDATE_POOL_SIZE, recentlyLogged);
    if (candidates.length === 0) {
      return { recommendations: [], modelVersion: 'none' };
    }
  }

  const mlUserId = stablePositiveInt(userId);
  const requestId = `sf-${userId}-${Date.now()}`;
  const mealByMlId = new Map<number, MealCandidate>();
  const recommendationIdByMealId = new Map<string, string>();
  const instances: MLInstance[] = candidates.map(meal => {
    const mlRecipeId = stablePositiveInt(meal.meal_id);
    const recommendationId = randomUUID();
    mealByMlId.set(mlRecipeId, meal);
    recommendationIdByMealId.set(meal.meal_id, recommendationId);
    return {
      user_id: mlUserId,
      recipe_id: mlRecipeId,
      recommendation_id: recommendationId,
      ...buildFeatureVector(meal, userFeatures),
    };
  });

  let ranked: Array<{ meal: MealCandidate; score: number }>;
  let modelVersion: string;

  try {
    const mlResponse = await callMLPredict(requestId, instances);
    ranked = mlResponse.predictions
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(p => ({ score: p.score, meal: mealByMlId.get(Number(p.recipe_id))! }))
      .filter((p): p is { meal: MealCandidate; score: number } => p.meal != null);
    modelVersion = mlResponse.model_version ?? 'unknown';
  } catch (err) {
    const reason = isColdStart ? 'cold-start' : 'ml-unavailable';
    log('warn', `[RecommendationService] ML API unavailable (${reason}), using macro-proximity ranking`, err);
    ranked = rankByMacroProximity(candidates, userFeatures, limit);
    modelVersion = reason;
  }

  const bucketed = assignBuckets(ranked, userFeatures, ranked.length);

  const saved = await recommendationRepository.saveRecommendations(
    userId,
    bucketed.map(r => ({
      recommendation_id: recommendationIdByMealId.get(r.meal.meal_id) ?? randomUUID(),
      request_id: requestId,
      meal_id: r.meal.meal_id,
      score: r.score,
      model_version: modelVersion,
      ml_user_id: mlUserId,
      ml_recipe_id: stablePositiveInt(r.meal.meal_id),
    }))
  );

  const savedById = new Map(saved.map(s => [s.meal_id, s]));

  const recommendations = bucketed.map(r => {
    const savedRec = savedById.get(r.meal.meal_id);
    return {
      recommendation_id: savedRec?.id ?? recommendationIdByMealId.get(r.meal.meal_id) ?? randomUUID(),
      meal_id: r.meal.meal_id,
      meal_name: r.meal.meal_name,
      description: r.meal.description,
      score: r.score,
      bucket: r.bucket,
      calories: r.meal.calories,
      protein_g: r.meal.protein,
      carbs_g: r.meal.carbs,
      fat_g: r.meal.fat,
      serving_size: r.meal.serving_size,
      serving_unit: r.meal.serving_unit,
      reason: reasonForRecommendation(r.meal, userFeatures),
    };
  });

  return { recommendations, modelVersion };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function recordFeedback(userId: any, recommendationId: string, action: string): Promise<void> {
  await recommendationRepository.logInteraction(userId, recommendationId, action);
  const context = await recommendationRepository.getRecommendationForFeedback(userId, recommendationId);
  if (!context?.request_id || context.ml_user_id == null || context.ml_recipe_id == null) {
    log('warn', '[RecommendationService] Feedback saved locally but not forwarded to ML; missing context', {
      recommendationId,
    });
    return;
  }
  try {
    await sendMLFeedback({
      request_id: context.request_id,
      recommendation_id: recommendationId,
      user_id: context.ml_user_id,
      recipe_id: context.ml_recipe_id,
      action,
    });
  } catch (err) {
    log('warn', '[RecommendationService] ML feedback forwarding failed; local feedback was retained', err);
  }
}

export default { getRecommendations, recordFeedback };
