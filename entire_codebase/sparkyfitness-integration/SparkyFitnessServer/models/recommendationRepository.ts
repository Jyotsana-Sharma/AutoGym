import { getClient } from '../db/poolManager.js';
import { log } from '../config/logging.js';

export interface MealCandidate {
  meal_id: string;
  meal_name: string;
  description: string | null;
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

export interface SavedRecommendation {
  id: string;
  meal_id: string;
  score: number;
  model_version: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getUserGoals(userId: any): Promise<UserGoals> {
  const client = await getClient(userId);
  try {
    const result = await client.query(
      `SELECT calories, protein, carbs, fat
         FROM user_goals
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1`
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
          AND entry_date >= CURRENT_DATE - ($1 || ' days')::interval`,
      [days]
    );
    return new Set(result.rows.map((r: any) => r.meal_template_id as string));
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
         WHERE (m.is_public = TRUE OR m.user_id = $1)
           ${excludeClause}
         GROUP BY m.id, m.name, m.description, m.serving_size, m.serving_unit
         ORDER BY RANDOM()
         LIMIT ${limit}`,
      [userId, ...excluded]
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return result.rows.map((r: any) => ({
      meal_id: r.meal_id,
      meal_name: r.meal_name,
      description: r.description ?? null,
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
async function saveRecommendations(userId: any, recommendations: Array<{ meal_id: string; score: number; model_version: string }>): Promise<SavedRecommendation[]> {
  if (recommendations.length === 0) return [];
  const client = await getClient(userId);
  try {
    // Ensure table exists
    await client.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto`);
    await client.query(`
      CREATE TABLE IF NOT EXISTS recommendation_cache (
        id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id       TEXT NOT NULL,
        meal_id       UUID NOT NULL,
        score         DOUBLE PRECISION NOT NULL,
        model_version TEXT NOT NULL DEFAULT 'unknown',
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at    TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 day')
      )
    `);

    const params: unknown[] = [];
    const valueClauses = recommendations.map((rec, i) => {
      const base = i * 4;
      params.push(userId, rec.meal_id, rec.score, rec.model_version);
      return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4})`;
    });

    const result = await client.query(
      `INSERT INTO recommendation_cache (user_id, meal_id, score, model_version)
       VALUES ${valueClauses.join(', ')}
       RETURNING id, meal_id, score, model_version`,
      params
    );

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return result.rows.map((r: any) => ({
      id: r.id,
      meal_id: r.meal_id,
      score: Number(r.score),
      model_version: r.model_version,
    }));
  } finally {
    client.release();
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function logInteraction(userId: any, recommendationId: string, action: string): Promise<void> {
  const client = await getClient(userId);
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS recommendation_interactions (
        id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        recommendation_id UUID NOT NULL,
        user_id           TEXT NOT NULL,
        action            TEXT NOT NULL,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )
    `);
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
  getCandidateMeals,
  saveRecommendations,
  logInteraction,
};
