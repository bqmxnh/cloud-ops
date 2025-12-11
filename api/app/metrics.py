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
FB_COUNT     = Counter("feedback_requests_total", "Feedback requests")
DRIFT_COUNT  = Counter("drift_events_total", "Number of drift detections")
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

    # Reload count đã được tăng từ model_loader
    MODEL_RELOAD_COUNT.inc(0)  

    return PlainTextResponse(generate_latest(), media_type="text/plain")


metrics_router = router
