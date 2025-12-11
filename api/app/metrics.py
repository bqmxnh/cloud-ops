from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

router = APIRouter()

PRED_COUNT = Counter("prediction_requests_total", "Prediction requests")
FB_COUNT = Counter("feedback_requests_total", "Feedback requests")
DRIFT_COUNT = Counter("drift_events_total", "Drift detected total")
LATENCY = Histogram("prediction_latency_ms", "Prediction latency (ms)")

@router.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

metrics_router = router