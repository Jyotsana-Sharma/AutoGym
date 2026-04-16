/**
 * RecipeRecommendations.tsx
 *
 * Personalized meal recommendation panel.
 * Uses the ML-powered /api/recommendations endpoint (SparkyFitness ML system).
 *
 * Usage: drop this component inside the Foods page or Meal Management page,
 * e.g. as a tab or sidebar section:
 *
 *   import RecipeRecommendations from "./RecipeRecommendations";
 *   <RecipeRecommendations />
 */

import { useCallback, useState } from "react";
import {
  useRecommendations,
  useRecommendationFeedback,
  useInvalidateRecommendations,
  RecommendedMeal,
} from "@/hooks/Foods/useRecommendations";

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function NutrientBadge({
  label,
  value,
  unit,
}: {
  label: string;
  value: number | null;
  unit: string;
}) {
  if (value == null) return null;
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
      <span className="font-medium">{label}</span>
      {value.toFixed(0)} {unit}
    </span>
  );
}

function MealCard({
  rec,
  onLog,
  onDismiss,
  onSave,
}: {
  rec: RecommendedMeal;
  onLog: (rec: RecommendedMeal) => void;
  onDismiss: (rec: RecommendedMeal) => void;
  onSave: (rec: RecommendedMeal) => void;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm transition hover:shadow-md">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-gray-900 leading-snug">
            {rec.meal_name}
          </h3>
          {rec.description && (
            <p className="mt-0.5 text-xs text-gray-500 line-clamp-2">
              {rec.description}
            </p>
          )}
        </div>
        {/* Score pill */}
        {rec.score > 0 && (
          <span
            className="shrink-0 rounded-full px-2 py-0.5 text-xs font-medium"
            style={{
              background: `hsl(${Math.round(rec.score * 120)}, 60%, 90%)`,
              color: `hsl(${Math.round(rec.score * 120)}, 60%, 30%)`,
            }}
          >
            {Math.round(rec.score * 100)}%
          </span>
        )}
      </div>

      {/* Nutrition badges */}
      <div className="flex flex-wrap gap-1.5">
        <NutrientBadge label="kcal" value={rec.calories} unit="" />
        <NutrientBadge label="P" value={rec.protein_g} unit="g" />
        <NutrientBadge label="C" value={rec.carbs_g} unit="g" />
        <NutrientBadge label="F" value={rec.fat_g} unit="g" />
      </div>

      {/* Recommendation reason */}
      <p className="text-xs italic text-indigo-600">{rec.reason}</p>

      {/* Actions */}
      <div className="flex gap-2 pt-1">
        <button
          onClick={() => onLog(rec)}
          className="flex-1 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          Add to Diary
        </button>
        <button
          onClick={() => onSave(rec)}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50"
          title="Save for later"
        >
          ★
        </button>
        <button
          onClick={() => onDismiss(rec)}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-xs text-gray-400 hover:bg-gray-50"
          title="Dismiss"
        >
          ✕
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface Props {
  /** Optionally restrict recommendations to a meal type */
  mealType?: "breakfast" | "lunch" | "dinner" | "snack" | "any";
  /** Max number of recommendations to show */
  limit?: number;
  /**
   * Called when the user clicks "Add to Diary".
   * Implement this to call SparkyFitness's existing log-meal endpoint.
   */
  onLogMeal?: (mealId: string) => void;
}

export default function RecipeRecommendations({
  mealType = "any",
  limit = 6,
  onLogMeal,
}: Props) {
  const invalidateRecommendations = useInvalidateRecommendations();
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  // ── Fetch recommendations ──────────────────────────────────────────────────
  const { data, isLoading, isError, refetch } = useRecommendations(mealType, limit);

  // ── Feedback mutation ──────────────────────────────────────────────────────
  const feedbackMutation = useRecommendationFeedback();

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleLog = useCallback(
    (rec: RecommendedMeal) => {
      feedbackMutation.mutate({ id: rec.recommendation_id, action: "logged" });
      onLogMeal?.(rec.meal_id);
    },
    [feedbackMutation, onLogMeal]
  );

  const handleDismiss = useCallback(
    (rec: RecommendedMeal) => {
      feedbackMutation.mutate({
        id: rec.recommendation_id,
        action: "dismissed",
      });
      setDismissed((prev) => new Set([...prev, rec.recommendation_id]));
    },
    [feedbackMutation]
  );

  const handleSave = useCallback(
    (rec: RecommendedMeal) => {
      feedbackMutation.mutate({ id: rec.recommendation_id, action: "saved" });
    },
    [feedbackMutation]
  );

  // ── Filter out dismissed ───────────────────────────────────────────────────
  const visible =
    data?.recommendations
      .filter((r) => !dismissed.has(r.recommendation_id))
      .slice(0, limit) ?? [];

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <section aria-labelledby="rec-heading" className="w-full">
      {/* Section header */}
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2
            id="rec-heading"
            className="text-base font-semibold text-gray-900"
          >
            Recommended for You
          </h2>
          <p className="mt-0.5 text-xs text-gray-500">
            Personalised suggestions based on your goals
          </p>
        </div>
        <button
          onClick={() => {
            setDismissed(new Set());
            invalidateRecommendations();
          }}
          className="text-xs text-indigo-600 hover:underline"
        >
          Refresh
        </button>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: limit }).map((_, i) => (
            <div
              key={i}
              className="h-40 animate-pulse rounded-xl bg-gray-100"
            />
          ))}
        </div>
      )}

      {/* Error state */}
      {isError && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Could not load recommendations.{" "}
          <button
            onClick={() => refetch()}
            className="font-medium underline hover:no-underline"
          >
            Try again
          </button>
        </div>
      )}

      {/* Empty state (after dismissals or no data) */}
      {!isLoading && !isError && visible.length === 0 && (
        <div className="rounded-xl border border-dashed border-gray-300 p-8 text-center text-sm text-gray-500">
          No recommendations right now.{" "}
          <button
            onClick={() => {
              setDismissed(new Set());
              refetch();
            }}
            className="text-indigo-600 hover:underline"
          >
            Reload
          </button>
        </div>
      )}

      {/* Cards */}
      {visible.length > 0 && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {visible.map((rec) => (
              <MealCard
                key={rec.recommendation_id}
                rec={rec}
                onLog={handleLog}
                onDismiss={handleDismiss}
                onSave={handleSave}
              />
            ))}
          </div>

          {/* Model attribution */}
          {data?.model_version && data.model_version !== "fallback" && (
            <p className="mt-3 text-right text-xs text-gray-400">
              Powered by SparkyFitness ML · model{" "}
              <span className="font-mono">{data.model_version}</span>
            </p>
          )}
        </>
      )}
    </section>
  );
}
