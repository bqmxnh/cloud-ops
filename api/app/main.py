from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import init_model
from app.websocket import router as ws_router
from app.inference import router as predict_router
from app.metrics import router as metrics_router
from app.evaluation import router as evaluate_router
from app.model_loader import init_model, auto_refresh_worker
from app import globals as G
import threading
import asyncio



app = FastAPI(title="IDS Drift Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_model()

    threading.Thread(
        target=auto_refresh_worker,
        daemon=True
    ).start()

    print("[STARTUP] Model auto-refresh worker started")


app.include_router(ws_router)
app.include_router(predict_router, prefix="/predict", tags=["Predict"])
app.include_router(metrics_router, tags=["Metrics"])
app.include_router(evaluate_router, prefix="/evaluate", tags=["Evaluate"])

@app.get("/")
def root():
    return {
        "status": "running",
        "version": "9.0",
    }
