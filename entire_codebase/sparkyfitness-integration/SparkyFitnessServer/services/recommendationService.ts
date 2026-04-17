import { log } from '../config/logging.js';
import recommendationRepository, { MealCandidate, UserGoals } from '../models/recommendationRepository.js';

const ML_URL = process.env.ML_RECOMMENDATION_URL ?? 'http://localhost:8000';
const ML_MODEL_NAME = process.env.ML_MODEL_NAME ?? 'sparky-ranker';
const CANDIDATE_POOL_SIZE = 200;

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

function buildFeatureVector(meal: MealCandidate, goals: UserGoals): Record<string, number | string> {
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
    daily_calorie_target: goals.calories,
    protein_target_g: goals.protein,
    carbs_target_g: goals.carbs,
    fat_target_g: goals.fat,
    user_vegetarian: 0, user_vegan: 0, user_gluten_free: 0,
    user_dairy_free: 0, user_low_sodium: 0, user_low_fat: 0,
    history_pc1: 0, history_pc2: 0, history_pc3: 0,
    history_pc4: 0, history_pc5: 0, history_pc6: 0,
  };
}

function reasonForRecommendation(meal: MealCandidate, goals: UserGoals): string {
  const protein = meal.protein ?? 0;
  const cal = meal.calories ?? 0;
  const pct = goals.calories > 0 ? Math.round((cal / goals.calories) * 100) : 0;
  if (protein >= 25 && protein >= goals.protein * 0.3) {
    return `High protein (${protein.toFixed(0)} g) — fits your ${goals.protein} g target`;
  }
  if (cal > 0 && pct >= 20 && pct <= 35) {
    return `Balanced meal — ~${pct}% of your daily ${goals.calories} kcal goal`;
  }
  if (cal > 0 && cal <= goals.calories * 0.2) {
    return `Light option — only ${cal.toFixed(0)} kcal`;
  }
  return 'Matches your nutritional profile';
}

// ---------------------------------------------------------------------------
// ML API client
// ---------------------------------------------------------------------------
interface MLInstance extends Record<string, number | string> {
  user_id: string;
  recipe_id: string;
}

interface MLResponse {
  predictions: Array<{ recipe_id: string; score: number }>;
  model_version?: string;
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
// Exported service functions
// ---------------------------------------------------------------------------
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getRecommendations(userId: any, limit: number, excludeRecentDays: number) {
  const [goals, recentlyLogged] = await Promise.all([
    recommendationRepository.getUserGoals(userId),
    recommendationRepository.getRecentlyLoggedMealIds(userId, excludeRecentDays),
  ]);

  const candidates = await recommendationRepository.getCandidateMeals(userId, recentlyLogged, CANDIDATE_POOL_SIZE);
  if (candidates.length === 0) {
    return { recommendations: [], modelVersion: 'none' };
  }

  const instances: MLInstance[] = candidates.map(meal => ({
    user_id: String(userId),
    recipe_id: meal.meal_id,
    ...buildFeatureVector(meal, goals),
  }));

  let mlResponse: MLResponse;
  try {
    mlResponse = await callMLPredict(`sf-${userId}-${Date.now()}`, instances);
  } catch (err) {
    log('warn', '[RecommendationService] ML API unavailable, using fallback ranking', err);
    const fallback = [...candidates]
      .sort((a, b) => Math.abs((a.protein ?? 0) - goals.protein * 0.3) - Math.abs((b.protein ?? 0) - goals.protein * 0.3))
      .slice(0, limit);
    return {
      recommendations: fallback.map(m => ({
        recommendation_id: crypto.randomUUID(),
        meal_id: m.meal_id,
        meal_name: m.meal_name,
        description: m.description,
        score: 0,
        calories: m.calories,
        protein: m.protein,
        carbs: m.carbs,
        fat: m.fat,
        serving_size: m.serving_size,
        serving_unit: m.serving_unit,
        reason: reasonForRecommendation(m, goals),
      })),
      modelVersion: 'fallback',
    };
  }

  const mealById = new Map(candidates.map(m => [m.meal_id, m]));
  const ranked = mlResponse.predictions
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(p => ({ ...p, meal: mealById.get(p.recipe_id) }))
    .filter((p): p is typeof p & { meal: MealCandidate } => p.meal != null);

  const saved = await recommendationRepository.saveRecommendations(
    userId,
    ranked.map(r => ({ meal_id: r.recipe_id, score: r.score, model_version: mlResponse.model_version ?? 'unknown' }))
  );

  const savedById = new Map(saved.map(s => [s.meal_id, s]));
  const modelVersion = mlResponse.model_version ?? 'unknown';

  const recommendations = ranked.map(r => {
    const savedRec = savedById.get(r.recipe_id);
    return {
      recommendation_id: savedRec?.id ?? crypto.randomUUID(),
      meal_id: r.recipe_id,
      meal_name: r.meal.meal_name,
      description: r.meal.description,
      score: r.score,
      calories: r.meal.calories,
      protein: r.meal.protein,
      carbs: r.meal.carbs,
      fat: r.meal.fat,
      serving_size: r.meal.serving_size,
      serving_unit: r.meal.serving_unit,
      reason: reasonForRecommendation(r.meal, goals),
    };
  });

  return { recommendations, modelVersion };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function recordFeedback(userId: any, recommendationId: string, action: string): Promise<void> {
  await recommendationRepository.logInteraction(userId, recommendationId, action);
}

export default { getRecommendations, recordFeedback };
