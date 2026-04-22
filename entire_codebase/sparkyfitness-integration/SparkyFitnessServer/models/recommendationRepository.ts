import { getClient } from '../db/poolManager.js';
import { log } from '../config/logging.js';

export interface MealCandidate {
  meal_id: string;
  meal_name: string;
  description: string | null;
  ingredient_text: string | null;
  serving_size: number | null;
  serving_unit: string | null;
  calories: number | null;
  protein: number | null;
  carbs: number | null;
  fat: number | null;
  saturated_fat: number | null;
  sugars: number | null;
  sodium: number | null;
  dietary_fiber: number | null;
  n_ingredients: number;
}

export interface UserGoals {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface RecentLoggedMealHistory {
  text_documents: string[];
  ingredient_documents: string[];
  avg_calories: number;
  avg_protein_g: number;
  avg_carbohydrate_g: number;
  avg_total_fat_g: number;
}

export interface SavedRecommendation {
  id: string;
  request_id: string | null;
  meal_id: string;
  score: number;
  model_version: string;
  ml_user_id: number | null;
  ml_recipe_id: number | null;
}

export interface RecommendationFeedbackContext {
  recommendation_id: string;
  request_id: string | null;
  user_id: string;
  meal_id: string;
  ml_user_id: number | null;
  ml_recipe_id: number | null;
  model_version: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getUserGoals(userId: any): Promise<UserGoals> {
  const client = await getClient(userId);
  try {
    const result = await client.query(
      `SELECT calories, protein, carbs, fat
         FROM user_goals
        WHERE user_id = $1
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1`,
      [userId]
    );
    if (result.rows.length === 0) {
      return { calories: 2000, protein: 50, carbs: 250, fat: 65 };
    }
    const r = result.rows[0];
    return {
      calories: Number(r.calories) || 2000,
      protein: Number(r.protein) || 50,
      carbs: Number(r.carbs) || 250,
      fat: Number(r.fat) || 65,
    };
  } finally {
    client.release();
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getRecentlyLoggedMealIds(userId: any, days: number): Promise<Set<string>> {
  const client = await getClient(userId);
  try {
    const result = await client.query(
      `SELECT DISTINCT meal_template_id
         FROM food_entry_meals
        WHERE meal_template_id IS NOT NULL
          AND user_id = $1
          AND entry_date >= CURRENT_DATE - ($2 * interval '1 day')`,
      [userId, days]
    );
    return new Set(result.rows.map((r: any) => r.meal_template_id as string));
  } finally {
    client.release();
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getRecentLoggedMealHistory(userId: any, limit: number): Promise<RecentLoggedMealHistory> {
  const client = await getClient(userId);
  try {
    const result = await client.query(
      `WITH recent AS (
         SELECT entry_date, meal_template_id
           FROM food_entry_meals
          WHERE meal_template_id IS NOT NULL
            AND user_id = $1
          ORDER BY entry_date DESC
          LIMIT $2
       )
       SELECT
         recent.meal_template_id                                                  AS meal_id,
         m.name                                                                   AS meal_name,
         COALESCE(m.description, '')                                              AS description,
         COALESCE(STRING_AGG(LOWER(REGEXP_REPLACE(COALESCE(f.name, ''), '[^a-z0-9]+', ' ', 'g')), ' '), '') AS ingredient_text,
         COALESCE(SUM(fv.calories      * mf.quantity / NULLIF(fv.serving_size, 0)), 0) AS calories,
         COALESCE(SUM(fv.protein       * mf.quantity / NULLIF(fv.serving_size, 0)), 0) AS protein,
         COALESCE(SUM(fv.carbs         * mf.quantity / NULLIF(fv.serving_size, 0)), 0) AS carbs,
         COALESCE(SUM(fv.fat           * mf.quantity / NULLIF(fv.serving_size, 0)), 0) AS fat
       FROM recent
       JOIN meals m ON m.id = recent.meal_template_id
       LEFT JOIN meal_foods mf ON mf.meal_id = m.id
       LEFT JOIN food_variants fv ON fv.id = mf.variant_id
       LEFT JOIN foods f ON f.id = fv.food_id
       GROUP BY recent.entry_date, recent.meal_template_id, m.name, m.description
       ORDER BY recent.entry_date DESC`,
      [userId, limit]
    );

    if (result.rows.length === 0) {
      return {
        text_documents: [],
        ingredient_documents: [],
        avg_calories: 0,
        avg_protein_g: 0,
        avg_carbohydrate_g: 0,
        avg_total_fat_g: 0,
      };
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rows = result.rows.map((r: any) => ({
      text: [r.meal_name, r.description].filter(Boolean).join(' ').trim(),
      ingredient_text: (r.ingredient_text ?? '').trim(),
      calories: Number(r.calories) || 0,
      protein: Number(r.protein) || 0,
      carbs: Number(r.carbs) || 0,
      fat: Number(r.fat) || 0,
    }));

    const count = rows.length || 1;
    return {
      text_documents: rows.map(r => r.text).filter(Boolean),
      ingredient_documents: rows.map(r => r.ingredient_text).filter(Boolean),
      avg_calories: rows.reduce((sum, row) => sum + row.calories, 0) / count,
      avg_protein_g: rows.reduce((sum, row) => sum + row.protein, 0) / count,
      avg_carbohydrate_g: rows.reduce((sum, row) => sum + row.carbs, 0) / count,
      avg_total_fat_g: rows.reduce((sum, row) => sum + row.fat, 0) / count,
    };
  } finally {
    client.release();
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getCandidateMeals(userId: any, excludeMealIds: Set<string>, limit: number): Promise<MealCandidate[]> {
  const client = await getClient(userId);
  try {
    // Build exclusion clause
    const excluded = excludeMealIds.size > 0 ? [...excludeMealIds] : [];
    const excludeClause = excluded.length > 0
      ? `AND m.id NOT IN (${excluded.map((_: any, i: number) => `$${i + 2}`).join(',')})`
      : '';

    const result = await client.query(
      `SELECT
           m.id                                                                 AS meal_id,
           m.name                                                               AS meal_name,
           m.description,
           COALESCE(STRING_AGG(LOWER(REGEXP_REPLACE(COALESCE(f.name, ''), '[^a-z0-9]+', ' ', 'g')), ' '), '') AS ingredient_text,
           m.serving_size,
           m.serving_unit,
           COUNT(mf.id)                                                         AS n_ingredients,
           SUM(fv.calories      * mf.quantity / NULLIF(fv.serving_size, 0))    AS calories,
           SUM(fv.protein       * mf.quantity / NULLIF(fv.serving_size, 0))    AS protein,
           SUM(fv.carbs         * mf.quantity / NULLIF(fv.serving_size, 0))    AS carbs,
           SUM(fv.fat           * mf.quantity / NULLIF(fv.serving_size, 0))    AS fat,
           SUM(fv.saturated_fat * mf.quantity / NULLIF(fv.serving_size, 0))    AS saturated_fat,
           SUM(fv.sugars        * mf.quantity / NULLIF(fv.serving_size, 0))    AS sugars,
           SUM(fv.sodium        * mf.quantity / NULLIF(fv.serving_size, 0))    AS sodium,
           SUM(fv.dietary_fiber * mf.quantity / NULLIF(fv.serving_size, 0))    AS dietary_fiber
         FROM meals m
         LEFT JOIN meal_foods mf ON mf.meal_id = m.id
         LEFT JOIN food_variants fv ON fv.id = mf.variant_id
         LEFT JOIN foods f ON f.id = fv.food_id
         WHERE (m.is_public = TRUE OR m.user_id = $1)
           ${excludeClause}
         GROUP BY m.id, m.name, m.description, m.serving_size, m.serving_unit
         ORDER BY RANDOM()
         LIMIT ${limit}`,
      [userId, ...excluded]
    );

    if (result.rows.length > 0) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return result.rows.map((r: any) => ({
        meal_id: r.meal_id,
        meal_name: r.meal_name,
        description: r.description ?? null,
        ingredient_text: r.ingredient_text ?? null,
        serving_size: r.serving_size ? Number(r.serving_size) : null,
        serving_unit: r.serving_unit ?? null,
        n_ingredients: Number(r.n_ingredients),
        calories: r.calories ? Number(r.calories) : null,
        protein: r.protein ? Number(r.protein) : null,
        carbs: r.carbs ? Number(r.carbs) : null,
        fat: r.fat ? Number(r.fat) : null,
        saturated_fat: r.saturated_fat ? Number(r.saturated_fat) : null,
        sugars: r.sugars ? Number(r.sugars) : null,
        sodium: r.sodium ? Number(r.sodium) : null,
        dietary_fiber: r.dietary_fiber ? Number(r.dietary_fiber) : null,
      }));
    }

    const foodResult = await client.query(
      `SELECT DISTINCT ON (f.id)
           f.id             AS meal_id,
           f.name           AS meal_name,
           f.brand          AS description,
           LOWER(REGEXP_REPLACE(TRIM(CONCAT_WS(' ', f.name, f.brand)), '[^a-z0-9]+', ' ', 'g')) AS ingredient_text,
           fv.serving_size,
           fv.serving_unit,
           1                AS n_ingredients,
           fv.calories,
           fv.protein,
           fv.carbs,
           fv.fat,
           fv.saturated_fat,
           fv.sugars,
           fv.sodium,
           fv.dietary_fiber
         FROM foods f
         JOIN food_variants fv ON fv.food_id = f.id
         WHERE f.user_id = $1 OR f.shared_with_public = TRUE
         ORDER BY f.id, fv.is_default DESC, fv.updated_at DESC
         LIMIT $2`,
      [userId, limit]
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return foodResult.rows.map((r: any) => ({
      meal_id: r.meal_id,
      meal_name: r.meal_name,
      description: r.description ?? null,
      ingredient_text: r.ingredient_text ?? null,
      serving_size: r.serving_size ? Number(r.serving_size) : null,
      serving_unit: r.serving_unit ?? null,
      n_ingredients: Number(r.n_ingredients),
      calories: r.calories ? Number(r.calories) : null,
      protein: r.protein ? Number(r.protein) : null,
      carbs: r.carbs ? Number(r.carbs) : null,
      fat: r.fat ? Number(r.fat) : null,
      saturated_fat: r.saturated_fat ? Number(r.saturated_fat) : null,
      sugars: r.sugars ? Number(r.sugars) : null,
      sodium: r.sodium ? Number(r.sodium) : null,
      dietary_fiber: r.dietary_fiber ? Number(r.dietary_fiber) : null,
    }));
  } finally {
    client.release();
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function saveRecommendations(
  userId: any,
  recommendations: Array<{
    recommendation_id: string;
    request_id: string;
    meal_id: string;
    score: number;
    model_version: string;
    ml_user_id: number;
    ml_recipe_id: number;
  }>
): Promise<SavedRecommendation[]> {
  if (recommendations.length === 0) return [];
  const client = await getClient(userId);
  try {
    const params: unknown[] = [];
    const valueClauses = recommendations.map((rec, i) => {
      const base = i * 8;
      params.push(
        rec.recommendation_id,
        rec.request_id,
        userId,
        rec.meal_id,
        rec.ml_user_id,
        rec.ml_recipe_id,
        rec.score,
        rec.model_version
      );
      return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4}, $${base + 5}, $${base + 6}, $${base + 7}, $${base + 8})`;
    });

    const result = await client.query(
      `INSERT INTO recommendation_cache
         (id, request_id, user_id, meal_id, ml_user_id, ml_recipe_id, score, model_version)
       VALUES ${valueClauses.join(', ')}
       ON CONFLICT (id) DO UPDATE SET
         request_id = EXCLUDED.request_id,
         score = EXCLUDED.score,
         model_version = EXCLUDED.model_version,
         expires_at = NOW() + INTERVAL '1 day'
       RETURNING id, request_id, meal_id, score, model_version, ml_user_id, ml_recipe_id`,
      params
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return result.rows.map((r: any) => ({
      id: r.id,
      request_id: r.request_id,
      meal_id: r.meal_id,
      score: Number(r.score),
      model_version: r.model_version,
      ml_user_id: r.ml_user_id == null ? null : Number(r.ml_user_id),
      ml_recipe_id: r.ml_recipe_id == null ? null : Number(r.ml_recipe_id),
    }));
  } finally {
    client.release();
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getRecommendationForFeedback(userId: any, recommendationId: string): Promise<RecommendationFeedbackContext | null> {
  const client = await getClient(userId);
  try {
    const result = await client.query(
      `SELECT id, request_id, user_id, meal_id, ml_user_id, ml_recipe_id, model_version
         FROM recommendation_cache
        WHERE id = $1 AND user_id = $2
        LIMIT 1`,
      [recommendationId, userId]
    );
    if (result.rows.length === 0) return null;
    const r = result.rows[0];
    return {
      recommendation_id: r.id,
      request_id: r.request_id,
      user_id: r.user_id,
      meal_id: r.meal_id,
      ml_user_id: r.ml_user_id == null ? null : Number(r.ml_user_id),
      ml_recipe_id: r.ml_recipe_id == null ? null : Number(r.ml_recipe_id),
      model_version: r.model_version,
    };
  } finally {
    client.release();
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function logInteraction(userId: any, recommendationId: string, action: string): Promise<void> {
  const client = await getClient(userId);
  try {
    await client.query(
      `INSERT INTO recommendation_interactions (recommendation_id, user_id, action)
       VALUES ($1, $2, $3)`,
      [recommendationId, userId, action]
    );
  } catch (err) {
    log('error', 'Failed to log recommendation interaction', err);
  } finally {
    client.release();
  }
}

export default {
  getUserGoals,
  getRecentlyLoggedMealIds,
  getRecentLoggedMealHistory,
  getCandidateMeals,
  saveRecommendations,
  getRecommendationForFeedback,
  logInteraction,
};
