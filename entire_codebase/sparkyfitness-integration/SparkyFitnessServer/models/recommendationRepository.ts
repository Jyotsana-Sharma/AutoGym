/**
 * recommendationRepository.ts
 *
 * Database access layer for the personalized recipe recommendation feature.
 * Follows the same pg-based pattern used throughout SparkyFitnessServer/models/.
 */

import { Pool } from "pg";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface MealCandidate {
  meal_id: string;
  meal_name: string;
  description: string | null;
  is_public: boolean;
  serving_size: number | null;
  serving_unit: string | null;
  // Aggregated nutritional totals (computed from meal_foods × food_variants)
  calories: number | null;
  protein_g: number | null;
  carbs_g: number | null;
  fat_g: number | null;
  saturated_fat_g: number | null;
  sugar_g: number | null;
  sodium_g: number | null;
  fiber_g: number | null;
  n_ingredients: number;
}

export interface UserGoals {
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
}

export interface SavedRecommendation {
  id: string;
  meal_id: string;
  score: number;
  model_version: string;
}

// ---------------------------------------------------------------------------
// Repository
// ---------------------------------------------------------------------------
export class RecommendationRepository {
  constructor(private readonly pool: Pool) {}

  /**
   * Fetch the user's latest nutritional goals.
   * Falls back to sensible defaults if no goals are configured.
   */
  async getUserGoals(userId: string): Promise<UserGoals> {
    const result = await this.pool.query(
      `SELECT
         COALESCE(calories, 2000)   AS calories,
         COALESCE(protein,  50)     AS protein_g,
         COALESCE(carbs,    250)    AS carbs_g,
         COALESCE(fat,      65)     AS fat_g
       FROM user_goals
       WHERE user_id = $1
       ORDER BY created_at DESC
       LIMIT 1`,
      [userId]
    );
    if (result.rows.length === 0) {
      return { calories: 2000, protein_g: 50, carbs_g: 250, fat_g: 65 };
    }
    const r = result.rows[0];
    return {
      calories: Number(r.calories),
      protein_g: Number(r.protein_g),
      carbs_g: Number(r.carbs_g),
      fat_g: Number(r.fat_g),
    };
  }

  /**
   * Return meal IDs the user logged recently (to avoid re-recommending).
   */
  async getRecentlyLoggedMealIds(
    userId: string,
    days: number
  ): Promise<Set<string>> {
    const result = await this.pool.query(
      `SELECT DISTINCT source_meal_id
         FROM food_entries
        WHERE user_id = $1
          AND source_meal_id IS NOT NULL
          AND date >= CURRENT_DATE - INTERVAL '1 day' * $2`,
      [userId, days]
    );
    return new Set(result.rows.map((r) => r.source_meal_id as string));
  }

  /**
   * Fetch candidate meals: the user's own meals + public meals.
   * Excludes recently logged meal IDs.
   * Computes per-meal nutritional totals from meal_foods × food_variants.
   */
  async getCandidateMeals(
    userId: string,
    excludeMealIds: Set<string>,
    limit: number
  ): Promise<MealCandidate[]> {
    const excluded =
      excludeMealIds.size > 0 ? [...excludeMealIds] : ["__no_exclusions__"];

    const result = await this.pool.query(
      `SELECT
           m.id                                         AS meal_id,
           m.name                                       AS meal_name,
           m.description,
           m.is_public,
           m.serving_size,
           m.serving_unit,
           COUNT(mf.food_id)                            AS n_ingredients,
           SUM(fv.calories     * mf.quantity / NULLIF(fv.serving_size,0)) AS calories,
           SUM(fv.protein      * mf.quantity / NULLIF(fv.serving_size,0)) AS protein_g,
           SUM(fv.carbs        * mf.quantity / NULLIF(fv.serving_size,0)) AS carbs_g,
           SUM(fv.fat          * mf.quantity / NULLIF(fv.serving_size,0)) AS fat_g,
           SUM(fv.saturated_fat* mf.quantity / NULLIF(fv.serving_size,0)) AS saturated_fat_g,
           SUM(fv.sugars       * mf.quantity / NULLIF(fv.serving_size,0)) AS sugar_g,
           SUM(fv.sodium       * mf.quantity / NULLIF(fv.serving_size,0)) AS sodium_g,
           SUM(fv.fiber        * mf.quantity / NULLIF(fv.serving_size,0)) AS fiber_g
         FROM meals m
         LEFT JOIN meal_foods mf ON mf.meal_id = m.id
         LEFT JOIN food_variants fv
                ON fv.food_id = mf.food_id
               AND fv.serving_unit = mf.unit
         WHERE (m.user_id = $1 OR m.is_public = TRUE)
           AND m.id NOT IN (${excluded.map((_, i) => `$${i + 3}`).join(",")})
         GROUP BY m.id, m.name, m.description, m.is_public, m.serving_size, m.serving_unit
         ORDER BY RANDOM()
         LIMIT $2`,
      [userId, limit, ...excluded]
    );

    return result.rows.map((r) => ({
      meal_id: r.meal_id,
      meal_name: r.meal_name,
      description: r.description ?? null,
      is_public: r.is_public,
      serving_size: r.serving_size ? Number(r.serving_size) : null,
      serving_unit: r.serving_unit ?? null,
      calories: r.calories ? Number(r.calories) : null,
      protein_g: r.protein_g ? Number(r.protein_g) : null,
      carbs_g: r.carbs_g ? Number(r.carbs_g) : null,
      fat_g: r.fat_g ? Number(r.fat_g) : null,
      saturated_fat_g: r.saturated_fat_g ? Number(r.saturated_fat_g) : null,
      sugar_g: r.sugar_g ? Number(r.sugar_g) : null,
      sodium_g: r.sodium_g ? Number(r.sodium_g) : null,
      fiber_g: r.fiber_g ? Number(r.fiber_g) : null,
      n_ingredients: Number(r.n_ingredients),
    }));
  }

  /**
   * Save a batch of recommendations to the cache table.
   * Returns the saved rows (with generated UUIDs).
   */
  async saveRecommendations(
    userId: string,
    recommendations: Array<{ meal_id: string; score: number; model_version: string }>
  ): Promise<SavedRecommendation[]> {
    if (recommendations.length === 0) return [];

    const values = recommendations
      .map(
        (_, i) =>
          `($${i * 3 + 1}, $${i * 3 + 2}, $${i * 3 + 3}, $${i * 3 + 4})`
      )
      .join(", ");

    // Flatten into a single params array: [userId, meal_id, score, model_version, ...]
    const params: unknown[] = [];
    for (const rec of recommendations) {
      params.push(userId, rec.meal_id, rec.score, rec.model_version);
    }

    const result = await this.pool.query(
      `INSERT INTO recommendation_cache (user_id, meal_id, score, model_version)
       VALUES ${recommendations
         .map((_, i) => `($${i * 4 + 1}, $${i * 4 + 2}, $${i * 4 + 3}, $${i * 4 + 4})`)
         .join(", ")}
       RETURNING id, meal_id, score, model_version`,
      params
    );

    return result.rows.map((r) => ({
      id: r.id,
      meal_id: r.meal_id,
      score: Number(r.score),
      model_version: r.model_version,
    }));
  }

  /**
   * Log a user interaction with a recommendation.
   */
  async logInteraction(
    recommendationId: string,
    userId: string,
    action: string
  ): Promise<void> {
    await this.pool.query(
      `INSERT INTO recommendation_interactions (recommendation_id, user_id, action)
       VALUES ($1, $2, $3)`,
      [recommendationId, userId, action]
    );
  }

  /**
   * Expire old recommendation cache entries (older than 24 hours).
   * Call periodically or on startup.
   */
  async pruneExpiredCache(): Promise<void> {
    await this.pool.query(
      `DELETE FROM recommendation_cache WHERE expires_at < NOW()`
    );
  }
}
