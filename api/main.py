"""
sa-fraud-shield FastAPI application.

Startup sequence:
    1. Load ML models into ModelRegistry (SIM swap + GNN)
    2. Register routers: /health, /score, /explain

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers.explain import router as explain_router
from api.routers.health import router as health_router
from api.routers.score import router as score_router
from api.services.model_registry import registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models…")
    registry.load_all(device="cpu")
    logger.info(f"Models ready — sim_swap={registry.sim_swap_loaded}  gnn={registry.gnn_loaded}")
    app.state.registry = registry
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="sa-fraud-shield",
    description="Real-time SA fraud detection API — SIM swap + GNN ensemble",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(score_router)
app.include_router(explain_router)


@app.get("/")
async def root():
    return {
        "service": "sa-fraud-shield",
        "version": "0.2.0",
        "models": {
            "sim_swap": registry.sim_swap_loaded,
            "gnn": registry.gnn_loaded,
        },
        "endpoints": ["/score", "/explain", "/health", "/docs"],
    }
