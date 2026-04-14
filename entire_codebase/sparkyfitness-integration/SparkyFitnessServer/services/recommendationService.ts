/**
 * recommendationService.ts
 *
 * Orchestrates personalized recipe recommendations by:
 *   1. Fetching user goals and candidate meals from SparkyFitness's DB
 *   2. Assembling the 44-feature vector expected by the ML serving API
 *   3. Calling the SparkyFitness ML service (POST /predict)
 *   4. Returning ranked meals with human-readable reason strings
 *
 * ML Serving API contract:
 *   POST {ML_RECOMMENDATION_URL}/predict
 *   Body: { request_id, model_name, instances: [{ user_id, recipe_id, ...features }] }
 *   Response: { predictions: [{ recipe_id, score }] }  (sorted by score desc)
 */

import { Pool } from "pg";
import {
  RecommendationRepository,
  MealCandidate,
  UserGoals,
} from "../models/recommendationRepository";
import { RecommendedMeal } from "../schemas/recommendationSchemas";

// ---------------------------------------------------------------------------
// Configuration (set via environment variables in .env / docker-compose)
// ---------------------------------------------------------------------------
const ML_URL =
  process.env.ML_RECOMMENDATION_URL ?? "http://localhost:8000";
const ML_MODEL_NAME =
  process.env.ML_MODEL_NAME ?? "sparky-ranker";
const CANDIDATE_POOL_SIZE = 200; // fetch up to this many meals; model ranks them

// ---------------------------------------------------------------------------
// Feature engineering helpers
// ---------------------------------------------------------------------------

/** % Daily Value reference amounts (FDA) */
const PDV_REF = {
  calories: 2000,
  fat: 78,         // g
  sugar: 50,       // g
  sodium: 2.3,     // g  (= 2300 mg)
  protein: 50,     // g
  saturated_fat: 20, // g
  carbohydrate: 275, // g
};

function toPDV(value: number | null, ref: number): number {
  if (value == null || ref === 0) return 0;
  return (value / ref) * 100;
}

/**
 * Infer broad dietary flags from a user's recent food entry history.
 * This is a best-effort heuristic; expand with explicit preference settings later.
 */
async function inferDietaryFlags(
  pool: Pool,
  userId: string
): Promise<{
  user_vegetarian: number;
  user_vegan: number;
  user_gluten_free: number;
  user_dairy_free: number;
  user_low_sodium: number;
  user_low_fat: number;
}> {
  // Check if the user has set any goal-level dietary flags in a preferences table.
  // SparkyFitness does not have an allergen/diet flag table yet — default all to 0.
  // TODO: wire to a future user_dietary_preferences table.
  return {
    user_vegetarian: 0,
    user_vegan: 0,
    user_gluten_free: 0,
    user_dairy_free: 0,
    user_low_sodium: 0,
    user_low_fat: 0,
  };
}

/**
 * Build one feature-vector row per candidate meal.
 * Fields that SparkyFitness cannot provide (cuisine, allergens, PCA history)
 * are set to 0 / "unknown" so the model degrades gracefully.
 */
function buildFeatureVector(
  meal: MealCandidate,
  goals: UserGoals,
  dietaryFlags: ReturnType<typeof inferDietaryFlags> extends Promise<infer T>
    ? T
    : never,
  userId: string
): Record<string, number | string> {
  const cal = meal.calories ?? 0;
  const fat = meal.fat_g ?? 0;
  const sugar = meal.sugar_g ?? 0;
  const sodium_g = (meal.sodium_g ?? 0) / 1000; // convert mg → g
  const protein = meal.protein_g ?? 0;
  const sat_fat = meal.saturated_fat_g ?? 0;
  const carb = meal.carbs_g ?? 0;

  return {
    // ── Recipe metadata (reasonable defaults for SparkyFitness meals) ────────
    minutes: 30,
    n_ingredients: meal.n_ingredients,
    n_steps: 5,
    avg_rating: 0,
    n_reviews: 0,
    cuisine: "unknown",

    // ── Nutrition — % Daily Value ────────────────────────────────────────────
    calories: toPDV(cal, PDV_REF.calories),
    total_fat: toPDV(fat, PDV_REF.fat),
    sugar: toPDV(sugar, PDV_REF.sugar),
    sodium: toPDV(sodium_g, PDV_REF.sodium),
    protein: toPDV(protein, PDV_REF.protein),
    saturated_fat: toPDV(sat_fat, PDV_REF.saturated_fat),
    carbohydrate: toPDV(carb, PDV_REF.carbohydrate),

    // ── Nutrition — grams ────────────────────────────────────────────────────
    total_fat_g: fat,
    sugar_g: sugar,
    sodium_g: sodium_g,
    protein_g: protein,
    saturated_fat_g: sat_fat,
    carbohydrate_g: carb,

    // ── Allergen flags (0 = not flagged; expand with ingredient parsing later)
    has_egg: 0,
    has_fish: 0,
    has_milk: 0,
    has_nuts: 0,
    has_peanut: 0,
    has_sesame: 0,
    has_shellfish: 0,
    has_soy: 0,
    has_wheat: 0,

    // ── User goals ────────────────────────────────────────────────────────────
    daily_calorie_target: goals.calories,
    protein_target_g: goals.protein_g,
    carbs_target_g: goals.carbs_g,
    fat_target_g: goals.fat_g,

    // ── Dietary flags ─────────────────────────────────────────────────────────
    ...dietaryFlags,

    // ── PCA history embeddings (set to 0 until we compute them from food_entries)
    history_pc1: 0,
    history_pc2: 0,
    history_pc3: 0,
    history_pc4: 0,
    history_pc5: 0,
    history_pc6: 0,
  };
}

/**
 * Generate a short human-readable reason for recommending a meal.
 */
function reasonForRecommendation(
  meal: MealCandidate,
  goals: UserGoals
): string {
  const cal = meal.calories ?? 0;
  const protein = meal.protein_g ?? 0;
  const pct = goals.calories > 0 ? Math.round((cal / goals.calories) * 100) : 0;

  if (protein >= 25 && protein >= goals.protein_g * 0.3) {
    return `High protein (${protein.toFixed(0)} g) — fits your ${goals.protein_g} g target`;
  }
  if (cal > 0 && pct >= 20 && pct <= 35) {
    return `Balanced meal — ~${pct}% of your daily ${goals.calories} kcal goal`;
  }
  if (cal > 0 && cal <= goals.calories * 0.2) {
    return `Light option — only ${cal.toFixed(0)} kcal`;
  }
  return "Matches your nutritional profile";
}

// ---------------------------------------------------------------------------
// ML API client
// ---------------------------------------------------------------------------
interface MLPredictRequest {
  request_id: string;
  model_name: string;
  instances: Array<
    Record<string, number | string> & { user_id: string; recipe_id: string }
  >;
}

interface MLPredictResponse {
  predictions: Array<{ recipe_id: string; score: number }>;
  model_version?: string;
}

async function callMLPredict(
  body: MLPredictRequest
): Promise<MLPredictResponse> {
  const response = await fetch(`${ML_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(10_000), // 10 s timeout
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(
      `ML service error ${response.status}: ${text.slice(0, 200)}`
    );
  }

  return response.json() as Promise<MLPredictResponse>;
}

// ---------------------------------------------------------------------------
// Main service class
// ---------------------------------------------------------------------------
export class RecommendationService {
  private readonly repo: RecommendationRepository;

  constructor(private readonly pool: Pool) {
    this.repo = new RecommendationRepository(pool);
  }

  /**
   * Generate up to `limit` ranked meal recommendations for the user.
   */
  async getRecommendations(
    userId: string,
    limit: number,
    excludeRecentDays: number
  ): Promise<{ recommendations: RecommendedMeal[]; modelVersion: string }> {
    // 1. Fetch user context
    const [goals, recentlyLogged, dietaryFlags] = await Promise.all([
      this.repo.getUserGoals(userId),
      this.repo.getRecentlyLoggedMealIds(userId, excludeRecentDays),
      inferDietaryFlags(this.pool, userId),
    ]);

    // 2. Fetch candidate meals
    const candidates = await this.repo.getCandidateMeals(
      userId,
      recentlyLogged,
      CANDIDATE_POOL_SIZE
    );

    if (candidates.length === 0) {
      return { recommendations: [], modelVersion: "none" };
    }

    // 3. Build ML request
    const instances = candidates.map((meal) => ({
      user_id: userId,
      recipe_id: meal.meal_id,
      ...buildFeatureVector(meal, goals, dietaryFlags, userId),
    }));

    const requestId = `sf-${userId}-${Date.now()}`;
    let mlResponse: MLPredictResponse;

    try {
      mlResponse = await callMLPredict({
        request_id: requestId,
        model_name: ML_MODEL_NAME,
        instances: instances as MLPredictRequest["instances"],
      });
    } catch (err) {
      // Graceful degradation: return meals sorted by protein proximity to goal
      console.error("[RecommendationService] ML API unavailable:", err);
      const fallback = candidates
        .sort((a, b) => {
          const distA = Math.abs((a.protein_g ?? 0) - goals.protein_g * 0.3);
          const distB = Math.abs((b.protein_g ?? 0) - goals.protein_g * 0.3);
          return distA - distB;
        })
        .slice(0, limit);

      return {
        recommendations: fallback.map((m) => ({
          recommendation_id: crypto.randomUUID(),
          meal_id: m.meal_id,
          meal_name: m.meal_name,
          description: m.description,
          score: 0,
          calories: m.calories,
          protein_g: m.protein_g,
          carbs_g: m.carbs_g,
          fat_g: m.fat_g,
          serving_size: m.serving_size,
          serving_unit: m.serving_unit,
          reason: reasonForRecommendation(m, goals),
        })),
        modelVersion: "fallback",
      };
    }

    // 4. Sort by ML score and take top `limit`
    const mealById = new Map(candidates.map((m) => [m.meal_id, m]));
    const ranked = mlResponse.predictions
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((p) => ({ ...p, meal: mealById.get(p.recipe_id) }))
      .filter((p): p is typeof p & { meal: MealCandidate } => p.meal != null);

    // 5. Persist to recommendation_cache
    const saved = await this.repo.saveRecommendations(
      userId,
      ranked.map((r) => ({
        meal_id: r.recipe_id,
        score: r.score,
        model_version: mlResponse.model_version ?? "unknown",
      }))
    );

    // 6. Build response — join saved IDs back
    const savedById = new Map(saved.map((s) => [s.meal_id, s]));
    const modelVersion = mlResponse.model_version ?? "unknown";

    const recommendations: RecommendedMeal[] = ranked.map((r) => {
      const savedRec = savedById.get(r.recipe_id);
      return {
        recommendation_id: savedRec?.id ?? crypto.randomUUID(),
        meal_id: r.recipe_id,
        meal_name: r.meal.meal_name,
        description: r.meal.description,
        score: r.score,
        calories: r.meal.calories,
        protein_g: r.meal.protein_g,
        carbs_g: r.meal.carbs_g,
        fat_g: r.meal.fat_g,
        serving_size: r.meal.serving_size,
        serving_unit: r.meal.serving_unit,
        reason: reasonForRecommendation(r.meal, goals),
      };
    });

    return { recommendations, modelVersion };
  }

  /**
   * Record what the user did with a recommendation.
   */
  async recordFeedback(
    recommendationId: string,
    userId: string,
    action: string
  ): Promise<void> {
    await this.repo.logInteraction(recommendationId, userId, action);
  }
}
