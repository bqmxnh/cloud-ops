from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest
)
from app import globals as G

router = APIRouter()

# =============================
# BASIC METRICS
# =============================
PRED_COUNT   = Counter("prediction_requests_total", "Prediction requests")
LATENCY      = Histogram("prediction_latency_ms", "Prediction latency (ms)")

# =============================
# MODEL VERSION METRICS
# =============================
MODEL_VERSION = Gauge("model_version", "Current MLflow model version")
MODEL_RELOAD_COUNT = Counter("model_reload_total", "Number of model reloads")


@router.get("/metrics")
def metrics():
    # Update model version before exporting
    if G.current_model_version is not None:
        MODEL_VERSION.set(G.current_model_version)
        
    return PlainTextResponse(generate_latest(), media_type="text/plain")


metrics_router = router
