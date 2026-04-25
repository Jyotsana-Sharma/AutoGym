import { log } from '../config/logging.js';
import recommendationRepository, { DiaryProfile, MealCandidate, UserGoals } from '../models/recommendationRepository.js';
import { createHash, randomUUID } from 'crypto';

const ML_URL = process.env.ML_RECOMMENDATION_URL ?? 'http://localhost:8000';
const ML_MODEL_NAME = process.env.ML_MODEL_NAME ?? 'sparky-ranker';
const CANDIDATE_POOL_SIZE = 1000;
const HISTORY_LOOKBACK_DAYS = 30;
const MIN_SIMILARITY_SCORE = 0.08;

type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack' | 'any';

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

function normalizeText(value: string | null | undefined): string {
  return (value ?? '').toLowerCase().replace(/[^a-z0-9\s]/g, ' ');
}

function tokensFor(value: string | null | undefined): Set<string> {
  const stopwords = new Set(['with', 'and', 'the', 'for', 'from', 'this', 'that', 'easy', 'quick']);
  return new Set(
    normalizeText(value)
      .split(/\s+/)
      .filter(token => token.length >= 4 && !stopwords.has(token))
  );
}

function classifyCategory(meal: Pick<MealCandidate, 'meal_name' | 'description' | 'category_label'>): string {
  if (meal.category_label && meal.category_label !== 'other') return meal.category_label;
  const text = normalizeText(`${meal.meal_name} ${meal.description ?? ''}`);
  if (/\b(pancake|waffle|oat|cereal|granola|muffin|toast|bagel|breakfast)\b/.test(text)) return 'breakfast';
  if (/\b(apple|berry|blueberry|banana|orange|fruit)\b/.test(text)) return 'fruit';
  if (/\b(yogurt|cheese|milk|parfait)\b/.test(text)) return 'dairy';
  if (/\b(cake|pie|cookie|brownie|dessert|sweet)\b/.test(text)) return 'dessert';
  if (/\b(salmon|tuna|cod|fish|shrimp)\b/.test(text)) return 'seafood';
  if (/\b(chicken|turkey|tender|pork|beef|steak|meat|protein)\b/.test(text)) return 'protein';
  if (/\b(pasta|rice|bread|linguine|noodle|grain)\b/.test(text)) return 'grain';
  if (/\b(salad|vegetable|broccoli|spinach|pepper)\b/.test(text)) return 'vegetable';
  return 'other';
}

function proteinSource(meal: Pick<MealCandidate, 'meal_name' | 'description'>): string {
  const text = normalizeText(`${meal.meal_name} ${meal.description ?? ''}`);
  if (/\bpork\b/.test(text)) return 'pork';
  if (/\b(chicken|tender)\b/.test(text)) return 'chicken';
  if (/\b(salmon|tuna|cod|fish|shrimp)\b/.test(text)) return 'seafood';
  if (/\b(beef|steak)\b/.test(text)) return 'beef';
  if (/\b(yogurt|cheese|milk)\b/.test(text)) return 'dairy';
  return 'other';
}

function buildProfileCategoryCounts(profile: DiaryProfile): Map<string, number> {
  const counts = new Map<string, number>();
  for (const name of profile.names) {
    const category = classifyCategory({ meal_name: name, description: null, category_label: null });
    counts.set(category, (counts.get(category) ?? 0) + 1);
  }
  return counts;
}

function profileCategoryAffinity(meal: MealCandidate, profile: DiaryProfile): number {
  if (profile.names.length === 0) return 0;
  const category = classifyCategory(meal);
  const counts = buildProfileCategoryCounts(profile);
  return Math.min(1, (counts.get(category) ?? 0) / Math.max(profile.names.length, 1) * 3);
}

function tokenOverlapScore(meal: MealCandidate, profile: DiaryProfile): number {
  const historyTokens = tokensFor(profile.names.join(' '));
  if (historyTokens.size === 0) return 0;
  const candidateTokens = tokensFor(`${meal.meal_name} ${meal.description ?? ''}`);
  let overlap = 0;
  for (const token of candidateTokens) {
    if (historyTokens.has(token)) overlap += 1;
  }
  return Math.min(1, overlap / 3);
}

function sourceAffinityScore(meal: MealCandidate, profile: DiaryProfile): number {
  const source = proteinSource(meal);
  if (source === 'other' || profile.names.length === 0) return 0;
  let matches = 0;
  for (const name of profile.names) {
    if (proteinSource({ meal_name: name, description: null }) === source) matches += 1;
  }
  return Math.min(1, matches / Math.max(profile.names.length, 1) * 4);
}

function mealTypeFeature(mealType: MealType, target: MealType): number {
  return target === mealType ? 1 : 0;
}

function buildFeatureVector(
  meal: MealCandidate,
  goals: UserGoals,
  profile: DiaryProfile,
  mealType: MealType
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
    meal_type_breakfast: mealTypeFeature('breakfast', mealType),
    meal_type_lunch: mealTypeFeature('lunch', mealType),
    meal_type_dinner: mealTypeFeature('dinner', mealType),
    meal_type_snack: mealTypeFeature('snack', mealType),
    user_preference_sample_count: profile.sampleCount,
    user_vegetarian: 0, user_vegan: 0, user_gluten_free: 0,
    user_dairy_free: 0, user_low_sodium: 0, user_low_fat: 0,
    user_low_sugar: goals.carbs <= 150 ? 1 : 0,
    user_high_protein: goals.protein >= 100 ? 1 : 0,
    history_pc1: 0, history_pc2: 0, history_pc3: 0,
    history_pc4: 0, history_pc5: 0, history_pc6: 0,
    category_affinity_score: profileCategoryAffinity(meal, profile),
    candidate_name_text: meal.meal_name,
    candidate_description_text: meal.description ?? '',
    candidate_ingredient_text: meal.description ?? meal.meal_name,
    user_history_name_text: profile.names.join(' '),
    user_history_ingredient_text: profile.ingredientText,
    user_history_avg_calories: profile.avgCalories || goals.calories / 4,
    user_history_avg_protein_g: profile.avgProtein || goals.protein / 4,
    user_history_avg_carbohydrate_g: profile.avgCarbs || goals.carbs / 4,
    user_history_avg_total_fat_g: profile.avgFat || goals.fat / 4,
  };
}

function reasonForRecommendation(meal: MealCandidate, profile: DiaryProfile): string {
  const overlap = tokenOverlapScore(meal, profile);
  if (overlap >= 0.34) {
    return 'Similar to foods you logged recently';
  }
  if (profileCategoryAffinity(meal, profile) > 0) {
    return `Similar ${classifyCategory(meal)} option`;
  }
  return 'Similar to your recent diary';
}

function explanationForRecommendation(meal: MealCandidate, profile: DiaryProfile): string[] {
  const lines: string[] = [];
  const category = classifyCategory(meal);
  if (profile.names.length > 0 && profileCategoryAffinity(meal, profile) > 0) {
    lines.push(`Matches your recent ${category} pattern`);
  }
  if (tokenOverlapScore(meal, profile) > 0) {
    lines.push('Shares words or ingredients with recent diary items');
  }
  return lines;
}

function similarityScore(meal: MealCandidate, profile: DiaryProfile): number {
  if (profile.names.length === 0) return 0;
  const affinity = profileCategoryAffinity(meal, profile);
  const overlap = tokenOverlapScore(meal, profile);
  const source = sourceAffinityScore(meal, profile);
  return overlap * 0.55 + affinity * 0.35 + source * 0.1;
}

function rankedBySimilarity<T extends { meal: MealCandidate; similarity: number }>(ranked: T[], limit: number): T[] {
  const similar = ranked
    .filter(item => item.similarity >= MIN_SIMILARITY_SCORE)
    .sort((a, b) => b.similarity - a.similarity);
  return similar.slice(0, limit);
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
  // Keep the value inside signed 31-bit integer range for the Python serving API.
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
// Exported service functions
// ---------------------------------------------------------------------------
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getRecommendations(userId: any, limit: number, excludeRecentDays: number, mealType: MealType = 'any') {
  const [goals, recentlyLogged, profile] = await Promise.all([
    recommendationRepository.getUserGoals(userId).catch(() => ({ calories: 2000, protein: 50, carbs: 250, fat: 65 })),
    recommendationRepository.getRecentlyLoggedMealIds(userId, excludeRecentDays).catch(() => new Set<string>()),
    recommendationRepository.getRecentDiaryProfile(userId, HISTORY_LOOKBACK_DAYS).catch(() => ({
      sampleCount: 0,
      names: [],
      ingredientText: '',
      avgCalories: 0,
      avgProtein: 0,
      avgCarbs: 0,
      avgFat: 0,
      mealTypeCounts: {},
    })),
  ]);

  const candidates = await recommendationRepository.getCandidateMeals(userId, recentlyLogged, CANDIDATE_POOL_SIZE);
  if (candidates.length === 0) {
    return { recommendations: [], modelVersion: 'none' };
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
      ...buildFeatureVector(meal, goals, profile, mealType),
    };
  });

  let mlResponse: MLResponse;
  try {
    mlResponse = await callMLPredict(requestId, instances);
  } catch (err) {
    log('warn', '[RecommendationService] ML API unavailable, using fallback ranking', err);
    const fallback = rankedBySimilarity(
      candidates.map(meal => ({ meal, similarity: similarityScore(meal, profile) })),
      limit
    )
      .map(item => item.meal)
      .slice(0, limit);
    const saved = await recommendationRepository.saveRecommendations(
      userId,
      fallback.map((m, idx) => ({
        recommendation_id: recommendationIdByMealId.get(m.meal_id) ?? randomUUID(),
        request_id: requestId,
        meal_id: m.meal_id,
        score: Math.max(0, 1 - idx / Math.max(fallback.length, 1)),
        model_version: 'fallback',
        ml_user_id: mlUserId,
        ml_recipe_id: stablePositiveInt(m.meal_id),
      }))
    );
    const savedById = new Map(saved.map(s => [s.meal_id, s]));
    return {
      recommendations: fallback.map(m => ({
        recommendation_id: savedById.get(m.meal_id)?.id ?? recommendationIdByMealId.get(m.meal_id) ?? randomUUID(),
        meal_id: m.meal_id,
        meal_name: m.meal_name,
        description: m.description,
        score: 0,
        calories: m.calories,
        protein_g: m.protein,
        carbs_g: m.carbs,
        fat_g: m.fat,
        serving_size: m.serving_size,
        serving_unit: m.serving_unit,
        candidate_type: m.candidate_type,
        food_id: m.food_id,
        variant_id: m.variant_id,
        strategy: profile.sampleCount > 0 ? 'personalized' : 'cold_start',
        category_label: classifyCategory(m),
        reason: reasonForRecommendation(m, profile),
        explanation: explanationForRecommendation(m, profile),
      })),
      modelVersion: 'fallback',
    };
  }

  const ranked = rankedBySimilarity(
    mlResponse.predictions
      .map(p => {
        const meal = mealByMlId.get(Number(p.recipe_id));
        if (!meal) return null;
        return {
          ...p,
          meal,
          similarity: similarityScore(meal, profile),
        };
      })
      .filter((p): p is { recipe_id: number | string; score: number; meal: MealCandidate; similarity: number } => p != null),
    limit
  )
    .map(p => ({ ...p, meal: mealByMlId.get(Number(p.recipe_id)) }))
    .filter((p): p is typeof p & { meal: MealCandidate } => p.meal != null);

  const saved = await recommendationRepository.saveRecommendations(
    userId,
    ranked.map(r => ({
      recommendation_id: recommendationIdByMealId.get(r.meal.meal_id) ?? randomUUID(),
      request_id: requestId,
      meal_id: r.meal.meal_id,
      score: r.similarity,
      model_version: mlResponse.model_version ?? 'unknown',
      ml_user_id: mlUserId,
      ml_recipe_id: Number(r.recipe_id),
    }))
  );

  const savedById = new Map(saved.map(s => [s.meal_id, s]));
  const modelVersion = mlResponse.model_version ?? 'unknown';

  const recommendations = ranked.map(r => {
    const savedRec = savedById.get(r.meal.meal_id);
    return {
      recommendation_id: savedRec?.id ?? randomUUID(),
      meal_id: r.meal.meal_id,
      meal_name: r.meal.meal_name,
      description: r.meal.description,
      score: r.similarity,
      calories: r.meal.calories,
      protein_g: r.meal.protein,
      carbs_g: r.meal.carbs,
      fat_g: r.meal.fat,
      serving_size: r.meal.serving_size,
      serving_unit: r.meal.serving_unit,
      candidate_type: r.meal.candidate_type,
      food_id: r.meal.food_id,
      variant_id: r.meal.variant_id,
      strategy: profile.sampleCount > 0 ? 'personalized' : 'cold_start',
      category_label: classifyCategory(r.meal),
      reason: reasonForRecommendation(r.meal, profile),
      explanation: explanationForRecommendation(r.meal, profile),
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
