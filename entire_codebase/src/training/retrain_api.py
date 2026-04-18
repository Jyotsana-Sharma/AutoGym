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

from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import BaseModel
import uvicorn

from .retrain_pipeline import run_retraining
from .model_registry import promote_to_production, rollback_production, get_production_version_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SparkyFitness Retrain API", version="1.0.0")

retrain_jobs_total = Counter(
    "sparky_retrain_jobs_total",
    "Total retraining jobs by status and trigger reason",
    ["status", "reason"],
)
retrain_failures_total = Counter(
    "sparky_retrain_failures_total",
    "Total retraining jobs that failed",
)
last_retrain_timestamp = Gauge(
    "sparky_last_retrain_timestamp_seconds",
    "Unix timestamp of the most recent completed successful retraining job",
)
retrain_job_running = Gauge(
    "sparky_retrain_job_running",
    "1 when a retraining job is running, otherwise 0",
)
alert_rollbacks_total = Counter(
    "sparky_alert_rollbacks_total",
    "Total rollback webhook requests by status",
    ["status"],
)

# ---------------------------------------------------------------------------
# In-memory job state (sufficient for single-node; swap with Redis for HA)
# ---------------------------------------------------------------------------
_job_lock = threading.Lock()
_current_job: dict[str, Any] = {"status": "idle", "result": None, "triggered_at": None}


class TriggerRequest(BaseModel):
    config: str = "configs/training/xgb_ranker.yaml"
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


class AlertRollbackRequest(BaseModel):
    alerts: list[dict[str, Any]] = []
    status: str | None = None
    commonLabels: dict[str, Any] = {}
    receiver: str | None = None


def _rollback_token() -> str:
    return os.environ.get("ROLLBACK_WEBHOOK_TOKEN", "")


def _allowed_rollback_alerts() -> set[str]:
    raw = os.environ.get("ROLLBACK_ALERT_NAMES", "HighErrorRate")
    return {name.strip() for name in raw.split(",") if name.strip()}


def _extract_firing_alert_names(payload: AlertRollbackRequest) -> set[str]:
    names: set[str] = set()
    for alert in payload.alerts:
        if alert.get("status", payload.status) not in {None, "firing"}:
            continue
        labels = alert.get("labels") or payload.commonLabels or {}
        if labels.get("severity") != "critical":
            continue
        alert_name = labels.get("alertname")
        if alert_name:
            names.add(str(alert_name))
    if not names and payload.status in {None, "firing"}:
        alert_name = payload.commonLabels.get("alertname")
        if alert_name and payload.commonLabels.get("severity") == "critical":
            names.add(str(alert_name))
    return names


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
        status = "completed" if result["success"] else "failed"
        retrain_jobs_total.labels(status=status, reason=req.reason).inc()
        retrain_job_running.set(0)
        if result["success"]:
            last_retrain_timestamp.set(datetime.now(timezone.utc).timestamp())
        else:
            retrain_failures_total.inc()
    except Exception as exc:
        logger.exception("Retraining job crashed: %s", exc)
        with _job_lock:
            _current_job["status"] = "failed"
            _current_job["result"] = {"error": str(exc)}
        retrain_jobs_total.labels(status="failed", reason=req.reason).inc()
        retrain_failures_total.inc()
        retrain_job_running.set(0)


@app.post("/trigger")
def trigger_retrain(req: TriggerRequest, background_tasks: BackgroundTasks):
    with _job_lock:
        if _current_job["status"] == "running":
            raise HTTPException(status_code=409, detail="A retraining job is already running")
        _current_job["status"] = "running"
        _current_job["triggered_at"] = datetime.now(timezone.utc).isoformat()
        _current_job["reason"] = req.reason
        _current_job["result"] = None

    retrain_jobs_total.labels(status="started", reason=req.reason).inc()
    retrain_job_running.set(1)
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


@app.post("/alerts/rollback")
def rollback_from_alert(
    req: AlertRollbackRequest,
    authorization: str | None = Header(default=None),
):
    """
    Authenticated webhook target for Grafana alerting.

    The endpoint is intentionally narrow: it only accepts Bearer-token requests
    for configured critical alert names, then rolls Production back to the
    previous archived MLflow Registry version.
    """
    expected_token = _rollback_token()
    if not expected_token:
        alert_rollbacks_total.labels(status="disabled").inc()
        raise HTTPException(status_code=403, detail="Rollback webhook token is not configured")

    if authorization != f"Bearer {expected_token}":
        alert_rollbacks_total.labels(status="unauthorized").inc()
        raise HTTPException(status_code=401, detail="Invalid rollback webhook token")

    firing_alerts = _extract_firing_alert_names(req)
    allowed_alerts = _allowed_rollback_alerts()
    actionable_alerts = sorted(firing_alerts & allowed_alerts)
    if not actionable_alerts:
        alert_rollbacks_total.labels(status="ignored").inc()
        return {
            "rolled_back": False,
            "reason": "No configured critical rollback alert is firing",
            "firing_alerts": sorted(firing_alerts),
            "allowed_alerts": sorted(allowed_alerts),
        }

    result = rollback_production(target_version=None)
    if not result.get("rolled_back"):
        alert_rollbacks_total.labels(status="failed").inc()
        raise HTTPException(status_code=400, detail=result.get("reason", "Rollback failed"))

    alert_rollbacks_total.labels(status="rolled_back").inc()
    result["triggered_by_alerts"] = actionable_alerts
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


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("RETRAIN_API_PORT", "8080")))
