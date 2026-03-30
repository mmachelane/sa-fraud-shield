"""
sa-fraud-shield FastAPI application.

Startup sequence:
    1. Load ML models into ModelRegistry (SIM swap + GNN)
    2. Build PSI drift detector baseline from training data
    3. Register routers: /health, /score, /explain
    4. Expose /metrics for Prometheus scraping

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.routers.explain import router as explain_router
from api.routers.health import router as health_router
from api.routers.score import router as score_router
from api.services.model_registry import registry
from monitoring.drift_detector import DriftDetector
from monitoring.metrics import MODEL_LOADED, PSI_SCORE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    registry.load_all(device="cpu")
    logger.info(f"Models ready — sim_swap={registry.sim_swap_loaded}  gnn={registry.gnn_loaded}")
    app.state.registry = registry

    # Record model health metrics
    MODEL_LOADED.labels(model="sim_swap").set(1 if registry.sim_swap_loaded else 0)
    MODEL_LOADED.labels(model="gnn").set(1 if registry.gnn_loaded else 0)

    # Build PSI drift detector baseline
    logger.info("Building PSI drift baseline...")
    drift_detector = DriftDetector.from_training_data(
        "data/processed/sim_swap_features.parquet",
        window_size=1000,
    )
    app.state.drift_detector = drift_detector
    logger.info("Startup complete.")

    # Background task: push PSI scores to Prometheus every 60s
    async def _update_psi_metrics():
        while True:
            await asyncio.sleep(60)
            try:
                report = drift_detector.report()
                for feat, psi_val in report.psi_scores.items():
                    PSI_SCORE.labels(feature=feat).set(psi_val)
                if report.drifted_features:
                    logger.warning(f"Drift detected: {report.drifted_features}")
            except Exception as e:
                logger.warning(f"PSI metric update failed: {e}")

    psi_task = asyncio.create_task(_update_psi_metrics())

    yield

    psi_task.cancel()
    logger.info("API shutting down")


app = FastAPI(
    title="sa-fraud-shield",
    description="Real-time SA fraud detection API — SIM swap + GNN ensemble",
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(score_router)
app.include_router(explain_router)


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/drift", tags=["monitoring"])
async def drift_report(request: Request):
    """PSI drift report for all monitored features."""
    detector: DriftDetector = request.app.state.drift_detector
    report = detector.report()
    return {
        "status": report.status,
        "n_live_samples": report.n_live_samples,
        "psi_scores": report.psi_scores,
        "drifted_features": report.drifted_features,
        "warned_features": report.warned_features,
        "thresholds": {"warn": 0.1, "alert": 0.2},
    }


@app.get("/")
async def root():
    return {
        "service": "sa-fraud-shield",
        "version": "0.3.0",
        "models": {
            "sim_swap": registry.sim_swap_loaded,
            "gnn": registry.gnn_loaded,
        },
        "endpoints": ["/score", "/explain", "/health", "/metrics", "/drift", "/docs"],
    }
