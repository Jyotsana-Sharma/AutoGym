/**
 * recommendations.ts — API client for the recommendation endpoints.
 * Mirrors the pattern used throughout SparkyFitnessFrontend/src/api/.
 */

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
  reason: string;
}

export interface RecommendationsResponse {
  recommendations: RecommendedMeal[];
  model_version: string;
  generated_at: string;
}

const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export async function fetchRecommendations(options?: {
  limit?: number;
  mealType?: "breakfast" | "lunch" | "dinner" | "snack" | "any";
  excludeRecentDays?: number;
}): Promise<RecommendationsResponse> {
  const params = new URLSearchParams();
  if (options?.limit) params.set("limit", String(options.limit));
  if (options?.mealType) params.set("meal_type", options.mealType);
  if (options?.excludeRecentDays !== undefined) {
    params.set("exclude_recent_days", String(options.excludeRecentDays));
  }

  const res = await fetch(
    `${BASE}/api/recommendations?${params.toString()}`,
    { credentials: "include" }
  );

  if (!res.ok) {
    throw new Error(`Recommendations API ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<RecommendationsResponse>;
}

export async function postRecommendationFeedback(
  recommendationId: string,
  action: "viewed" | "logged" | "dismissed" | "saved"
): Promise<void> {
  await fetch(`${BASE}/api/recommendations/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ recommendation_id: recommendationId, action }),
  });
}
