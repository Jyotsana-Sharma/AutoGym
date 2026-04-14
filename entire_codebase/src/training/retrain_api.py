"""Lightweight HTTP API that receives retraining webhook triggers.

Endpoints:
  POST /trigger    — start a retraining job (async background task)
  GET  /status     — status of the most recent retraining job
  POST /promote    — promote Staging → Production (manual approval gate)
  POST /rollback   — roll back Production to previous version
  GET  /health     — liveness check

Invoked by:
  - drift_monitor.py when drift exceeds threshold
  - scheduler (cron) for weekly forced retraining
  - GitHub Actions CD workflow
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

from .retrain_pipeline import run_retraining
from .model_registry import promote_to_production, rollback_production, get_production_version_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SparkyFitness Retrain API", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory job state (sufficient for single-node; swap with Redis for HA)
# ---------------------------------------------------------------------------
_job_lock = threading.Lock()
_current_job: dict[str, Any] = {"status": "idle", "result": None, "triggered_at": None}


class TriggerRequest(BaseModel):
    config: str = "configs/train/xgb_ranker.yaml"
    train_csv: str | None = None
    val_csv: str | None = None
    test_csv: str | None = None
    skip_data_checks: bool = False
    auto_promote: bool = False
    reason: str = "manual"


class PromoteRequest(BaseModel):
    version: str | None = None


class RollbackRequest(BaseModel):
    version: str | None = None


def _run_job(req: TriggerRequest):
    global _current_job
    try:
        result = run_retraining(
            config_path=req.config,
            train_csv=req.train_csv,
            val_csv=req.val_csv,
            test_csv=req.test_csv,
            skip_data_checks=req.skip_data_checks,
            auto_promote=req.auto_promote,
            model_export_path=os.environ.get("MODEL_EXPORT_PATH"),
        )
        with _job_lock:
            _current_job["status"] = "completed" if result["success"] else "failed"
            _current_job["result"] = result
    except Exception as exc:
        logger.exception("Retraining job crashed: %s", exc)
        with _job_lock:
            _current_job["status"] = "failed"
            _current_job["result"] = {"error": str(exc)}


@app.post("/trigger")
def trigger_retrain(req: TriggerRequest, background_tasks: BackgroundTasks):
    with _job_lock:
        if _current_job["status"] == "running":
            raise HTTPException(status_code=409, detail="A retraining job is already running")
        _current_job["status"] = "running"
        _current_job["triggered_at"] = datetime.now(timezone.utc).isoformat()
        _current_job["reason"] = req.reason
        _current_job["result"] = None

    background_tasks.add_task(_run_job, req)
    return {
        "accepted": True,
        "triggered_at": _current_job["triggered_at"],
        "reason": req.reason,
    }


@app.get("/status")
def get_status():
    with _job_lock:
        return dict(_current_job)


@app.post("/promote")
def promote(req: PromoteRequest):
    result = promote_to_production(version=req.version)
    if not result.get("promoted"):
        raise HTTPException(status_code=400, detail=result.get("reason", "Promotion failed"))
    return result


@app.post("/rollback")
def rollback(req: RollbackRequest):
    result = rollback_production(target_version=req.version)
    if not result.get("rolled_back"):
        raise HTTPException(status_code=400, detail=result.get("reason", "Rollback failed"))
    return result


@app.get("/model/production")
def production_model_info():
    info = get_production_version_info()
    if info is None:
        raise HTTPException(status_code=404, detail="No Production model registered")
    return info


@app.get("/health")
def health():
    return {"status": "healthy", "job_status": _current_job["status"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("RETRAIN_API_PORT", "8080")))
