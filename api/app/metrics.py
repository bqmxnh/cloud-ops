from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

router = APIRouter()

PREDICTION_COUNT = Counter("prediction_requests_total", "Prediction requests")
FEEDBACK_COUNT = Counter("feedback_requests_total", "Feedback requests")
LEARN_COUNT = Counter("model_learn_total", "Incremental learning updates")
LATENCY = Histogram("prediction_latency_ms", "Prediction latency (ms)")

@router.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

metrics_router = router
