import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchRecommendations,
  postRecommendationFeedback,
} from '@/api/recommendations';

export const useRecommendations = (
  mealType: 'breakfast' | 'lunch' | 'dinner' | 'snack' | 'any' = 'any',
  limit: number = 6
) => {
  return useQuery({
    queryKey: ['recommendations', mealType, limit],
    queryFn: () => fetchRecommendations({ limit: limit * 2, mealType }),
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
};

export const useRecommendationFeedback = () => {
  return useMutation({
    mutationFn: ({
      id,
      action,
    }: {
      id: string;
      action: 'viewed' | 'logged' | 'dismissed' | 'saved';
    }) => postRecommendationFeedback(id, action),
  });
};

export const useInvalidateRecommendations = () => {
  const queryClient = useQueryClient();
  return () =>
    queryClient.invalidateQueries({ queryKey: ['recommendations'] });
};
