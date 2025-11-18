from fastapi import FastAPI
from app.inference import router as inference_router
from app.feedback import router as feedback_router
from app.metrics import metrics_router

app = FastAPI(
    title="ARF IDS API",
    version="5.0-incremental-learning",
    description="Adaptive Random Forest IDS API with incremental learning"
)

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "IDS Cloud Ops API",
        "model": "Adaptive Random Forest",
        "version": "5.0"
    }

# Routers
app.include_router(inference_router, prefix="/predict", tags=["Predict"])
app.include_router(feedback_router, prefix="/feedback", tags=["Feedback"])
app.include_router(metrics_router, tags=["Metrics"])
