import { z } from "zod";

// ── Query params for GET /recommendations ────────────────────────────────────
export const GetRecommendationsQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(50).default(10),
  meal_type: z
    .enum(["breakfast", "lunch", "dinner", "snack", "any"])
    .default("any"),
  exclude_recent_days: z.coerce.number().int().min(0).max(30).default(7),
});

export type GetRecommendationsQuery = z.infer<
  typeof GetRecommendationsQuerySchema
>;

// ── Body for POST /recommendations/feedback ──────────────────────────────────
export const RecommendationFeedbackBodySchema = z.object({
  recommendation_id: z.string().uuid(),
  action: z.enum(["viewed", "logged", "dismissed", "saved"]),
});

export type RecommendationFeedbackBody = z.infer<
  typeof RecommendationFeedbackBodySchema
>;

// ── Outbound response shape ───────────────────────────────────────────────────
export const RecommendedMealSchema = z.object({
  recommendation_id: z.string().uuid(),
  meal_id: z.string(),
  meal_name: z.string(),
  description: z.string().nullable(),
  score: z.number(),
  calories: z.number().nullable(),
  protein_g: z.number().nullable(),
  carbs_g: z.number().nullable(),
  fat_g: z.number().nullable(),
  serving_size: z.number().nullable(),
  serving_unit: z.string().nullable(),
  candidate_type: z.enum(['meal', 'food']),
  food_id: z.string().nullable(),
  variant_id: z.string().nullable(),
  strategy: z.enum(['cold_start', 'personalized']),
  category_label: z.string(),
  reason: z.string(),
  explanation: z.array(z.string()),
});

export type RecommendedMeal = z.infer<typeof RecommendedMealSchema>;

export const RecommendationsResponseSchema = z.object({
  recommendations: z.array(RecommendedMealSchema),
  model_version: z.string(),
  generated_at: z.string(),
});
