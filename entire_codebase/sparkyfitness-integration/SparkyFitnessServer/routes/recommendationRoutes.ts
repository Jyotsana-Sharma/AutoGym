/**
 * recommendationRoutes.ts
 *
 * Express router for the personalized recipe recommendation feature.
 *
 * Mount in SparkyFitnessServer.ts:
 *   import recommendationRouter from "./routes/recommendationRoutes";
 *   app.use("/api", recommendationRouter);
 *
 * Endpoints:
 *   GET  /api/recommendations        — get ranked meal recommendations
 *   POST /api/recommendations/feedback — record user interaction
 */

import { Router, Request, Response, NextFunction } from "express";
import { Pool } from "pg";
import {
  GetRecommendationsQuerySchema,
  RecommendationFeedbackBodySchema,
} from "../schemas/recommendationSchemas";
import { RecommendationService } from "../services/recommendationService";

export function createRecommendationRouter(pool: Pool): Router {
  const router = Router();
  const service = new RecommendationService(pool);

  // ── GET /recommendations ─────────────────────────────────────────────────
  router.get(
    "/recommendations",
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        // Parse & validate query params
        const parsed = GetRecommendationsQuerySchema.safeParse(req.query);
        if (!parsed.success) {
          return res.status(400).json({ error: parsed.error.flatten() });
        }
        const { limit, exclude_recent_days } = parsed.data;

        // Resolve authenticated user ID.
        // Adapt to SparkyFitness's auth middleware — adjust the property name
        // to match what the session middleware sets (e.g. req.user?.id).
        const userId: string | undefined = (req as any).user?.id;
        if (!userId) {
          return res.status(401).json({ error: "Not authenticated" });
        }

        const { recommendations, modelVersion } =
          await service.getRecommendations(userId, limit, exclude_recent_days);

        return res.json({
          recommendations,
          model_version: modelVersion,
          generated_at: new Date().toISOString(),
        });
      } catch (err) {
        next(err);
      }
    }
  );

  // ── POST /recommendations/feedback ───────────────────────────────────────
  router.post(
    "/recommendations/feedback",
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const parsed = RecommendationFeedbackBodySchema.safeParse(req.body);
        if (!parsed.success) {
          return res.status(400).json({ error: parsed.error.flatten() });
        }
        const { recommendation_id, action } = parsed.data;

        const userId: string | undefined = (req as any).user?.id;
        if (!userId) {
          return res.status(401).json({ error: "Not authenticated" });
        }

        await service.recordFeedback(recommendation_id, userId, action);
        return res.json({ ok: true });
      } catch (err) {
        next(err);
      }
    }
  );

  return router;
}
