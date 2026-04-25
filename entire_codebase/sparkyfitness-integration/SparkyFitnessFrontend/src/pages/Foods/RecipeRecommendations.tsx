import { useCallback, useEffect, useRef, useState } from 'react';
import {
  useInvalidateRecommendations,
  useRecommendationFeedback,
  useRecommendations,
} from '@/hooks/Foods/useRecommendations';
import type { RecommendedMeal } from '@/hooks/Foods/useRecommendations';

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
  busy,
  onLog,
  onDismiss,
  onSave,
}: {
  rec: RecommendedMeal;
  busy: boolean;
  onLog: (rec: RecommendedMeal) => Promise<void>;
  onDismiss: (rec: RecommendedMeal) => Promise<void>;
  onSave: (rec: RecommendedMeal) => Promise<void>;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm transition hover:shadow-md">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="mb-1 flex flex-wrap gap-1">
            <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-[11px] font-medium text-indigo-700">
              {rec.category_label}
            </span>
            <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] text-gray-600">
              {rec.candidate_type === 'meal' ? 'Meal' : 'Food'}
            </span>
          </div>
          <h3 className="text-sm font-semibold leading-snug text-gray-900">
            {rec.meal_name}
          </h3>
          {rec.description && (
            <p className="mt-0.5 line-clamp-2 text-xs text-gray-500">
              {rec.description}
            </p>
          )}
        </div>
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

      <div className="flex flex-wrap gap-1.5">
        <NutrientBadge label="kcal" value={rec.calories} unit="" />
        <NutrientBadge label="P" value={rec.protein_g} unit="g" />
        <NutrientBadge label="C" value={rec.carbs_g} unit="g" />
        <NutrientBadge label="F" value={rec.fat_g} unit="g" />
      </div>

      <div className="space-y-1 text-xs text-gray-600">
        <p className="font-medium text-indigo-600">{rec.reason}</p>
        {rec.explanation?.slice(0, 2).map((line) => (
          <p key={line}>{line}</p>
        ))}
      </div>

      <div className="flex gap-2 pt-1">
        <button
          type="button"
          disabled={busy}
          onClick={() => onLog(rec)}
          className="flex-1 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-60"
        >
          Add to Diary
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => onSave(rec)}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50 disabled:opacity-60"
          title="Save for later"
        >
          Save
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => onDismiss(rec)}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-xs text-gray-400 hover:bg-gray-50 disabled:opacity-60"
          title="Dismiss"
        >
          Hide
        </button>
      </div>
    </div>
  );
}

interface Props {
  mealType?: 'breakfast' | 'lunch' | 'dinner' | 'snack' | 'any';
  limit?: number;
  onLogMeal?: (mealId: string) => void;
}

export default function RecipeRecommendations({
  mealType = 'any',
  limit = 6,
  onLogMeal,
}: Props) {
  const invalidateRecommendations = useInvalidateRecommendations();
  const feedbackMutation = useRecommendationFeedback();
  const viewedRecommendationIds = useRef<Set<string>>(new Set());
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [activeRecommendationId, setActiveRecommendationId] = useState<string | null>(null);
  const { data, isLoading, isError, refetch } = useRecommendations(mealType, limit);

  const handleLog = useCallback(
    async (rec: RecommendedMeal) => {
      setActiveRecommendationId(rec.recommendation_id);
      try {
        await feedbackMutation.mutateAsync({ id: rec.recommendation_id, action: 'logged' });
        setDismissed((prev) => new Set([...prev, rec.recommendation_id]));
        invalidateRecommendations();
        onLogMeal?.(rec.meal_id);
      } finally {
        setActiveRecommendationId(null);
      }
    },
    [feedbackMutation, invalidateRecommendations, onLogMeal]
  );

  const handleDismiss = useCallback(
    async (rec: RecommendedMeal) => {
      setActiveRecommendationId(rec.recommendation_id);
      try {
        await feedbackMutation.mutateAsync({ id: rec.recommendation_id, action: 'dismissed' });
        setDismissed((prev) => new Set([...prev, rec.recommendation_id]));
      } finally {
        setActiveRecommendationId(null);
      }
    },
    [feedbackMutation]
  );

  const handleSave = useCallback(
    async (rec: RecommendedMeal) => {
      setActiveRecommendationId(rec.recommendation_id);
      try {
        await feedbackMutation.mutateAsync({ id: rec.recommendation_id, action: 'saved' });
      } finally {
        setActiveRecommendationId(null);
      }
    },
    [feedbackMutation]
  );

  const visible =
    data?.recommendations
      .filter((recommendation) => !dismissed.has(recommendation.recommendation_id))
      .slice(0, limit) ?? [];

  useEffect(() => {
    for (const recommendation of visible) {
      if (viewedRecommendationIds.current.has(recommendation.recommendation_id)) continue;
      viewedRecommendationIds.current.add(recommendation.recommendation_id);
      feedbackMutation.mutate({ id: recommendation.recommendation_id, action: 'viewed' });
    }
  }, [feedbackMutation, visible]);

  return (
    <section aria-labelledby="rec-heading" className="w-full">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 id="rec-heading" className="text-base font-semibold text-gray-900">
            Recommended for You
          </h2>
          <p className="mt-0.5 text-xs text-gray-500">
            Personalised suggestions based on your goals and recent food history
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            setDismissed(new Set());
            invalidateRecommendations();
          }}
          className="text-xs text-indigo-600 hover:underline"
        >
          Refresh
        </button>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: limit }).map((_, i) => (
            <div key={i} className="h-40 animate-pulse rounded-xl bg-gray-100" />
          ))}
        </div>
      )}

      {isError && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Could not load recommendations.{' '}
          <button
            type="button"
            onClick={() => refetch()}
            className="font-medium underline hover:no-underline"
          >
            Try again
          </button>
        </div>
      )}

      {!isLoading && !isError && visible.length === 0 && (
        <div className="rounded-xl border border-dashed border-gray-300 p-8 text-center text-sm text-gray-500">
          No recommendations right now.{' '}
          <button
            type="button"
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

      {visible.length > 0 && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {visible.map((rec) => (
              <MealCard
                key={rec.recommendation_id}
                rec={rec}
                busy={activeRecommendationId === rec.recommendation_id}
                onLog={handleLog}
                onDismiss={handleDismiss}
                onSave={handleSave}
              />
            ))}
          </div>

          {data?.model_version && data.model_version !== 'fallback' && (
            <p className="mt-3 text-right text-xs text-gray-400">
              Powered by SparkyFitness ML model{' '}
              <span className="font-mono">{data.model_version}</span>
            </p>
          )}
        </>
      )}
    </section>
  );
}
