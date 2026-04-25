import express from 'express';
import { z } from 'zod';
import { log } from '../config/logging.js';
import recommendationService from '../services/recommendationService.js';

const router = express.Router();

const GetRecommendationsQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(50).default(10),
  meal_type: z.enum(['breakfast', 'lunch', 'dinner', 'snack', 'any']).default('any'),
  exclude_recent_days: z.coerce.number().int().min(0).max(30).default(7),
});

const FeedbackBodySchema = z.object({
  recommendation_id: z.string().uuid(),
  action: z.enum(['viewed', 'logged', 'dismissed', 'saved']),
});

/**
 * GET /api/recommendations
 * Returns personalised meal recommendations for the authenticated user.
 */
router.get('/', async (req, res) => {
  try {
    const parsed = GetRecommendationsQuerySchema.safeParse(req.query);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    const { limit, exclude_recent_days, meal_type } = parsed.data;
    const userId = req.userId;
    if (!userId) return res.status(401).json({ error: 'Not authenticated' });

    const { recommendations, modelVersion } = await recommendationService.getRecommendations(
      userId, limit, exclude_recent_days, meal_type
    );

    return res.json({
      recommendations,
      model_version: modelVersion,
      generated_at: new Date().toISOString(),
    });
  } catch (err) {
    log('error', 'GET /api/recommendations error', err);
    return res.status(500).json({ error: 'Failed to generate recommendations' });
  }
});

/**
 * POST /api/recommendations/feedback
 * Records what the user did with a recommendation (viewed/logged/dismissed/saved).
 */
router.post('/feedback', async (req, res) => {
  try {
    const parsed = FeedbackBodySchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    const userId = req.userId;
    if (!userId) return res.status(401).json({ error: 'Not authenticated' });

    await recommendationService.recordFeedback(userId, parsed.data.recommendation_id, parsed.data.action);
    return res.json({ ok: true });
  } catch (err) {
    log('error', 'POST /api/recommendations/feedback error', err);
    return res.status(500).json({ error: 'Failed to record feedback' });
  }
});

export default router;
