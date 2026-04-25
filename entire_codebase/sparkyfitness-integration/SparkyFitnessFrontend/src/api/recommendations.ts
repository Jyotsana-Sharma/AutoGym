/**
 * recommendations.ts — API client for the recommendation endpoints.
 * Mirrors the pattern used throughout SparkyFitnessFrontend/src/api/.
 */

import { apiCall } from '@/api/api';

export interface RecommendedMeal {
  recommendation_id: string;
  meal_id: string;
  meal_name: string;
  description: string | null;
  score: number;
  calories: number | null;
  protein_g: number | null;
  carbs_g: number | null;
  fat_g: number | null;
  serving_size: number | null;
  serving_unit: string | null;
  candidate_type: 'meal' | 'food';
  food_id: string | null;
  variant_id: string | null;
  strategy: 'cold_start' | 'personalized';
  category_label: string;
  reason: string;
  explanation: string[];
}

export interface RecommendationsResponse {
  recommendations: RecommendedMeal[];
  model_version: string;
  generated_at: string;
}

export async function fetchRecommendations(options?: {
  limit?: number;
  mealType?: "breakfast" | "lunch" | "dinner" | "snack" | "any";
  excludeRecentDays?: number;
}): Promise<RecommendationsResponse> {
  return apiCall('/recommendations', {
    method: 'GET',
    params: {
      ...(options?.limit !== undefined && { limit: String(options.limit) }),
      ...(options?.mealType && { meal_type: options.mealType }),
      ...(options?.excludeRecentDays !== undefined && {
        exclude_recent_days: String(options.excludeRecentDays),
      }),
    },
  });
}

export async function postRecommendationFeedback(
  recommendationId: string,
  action: "viewed" | "logged" | "dismissed" | "saved"
): Promise<void> {
  await apiCall('/recommendations/feedback', {
    method: 'POST',
    body: JSON.stringify({ recommendation_id: recommendationId, action }),
  });
}
