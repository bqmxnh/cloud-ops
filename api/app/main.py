from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import init_model
from app.websocket import router as ws_router
from app.inference import router as predict_router
from app.feedback import router as feedback_router
from app.metrics import router as metrics_router
from app import globals as G
import asyncio



app = FastAPI(title="IDS Drift Detector – v9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_model()
    # Start cleanup task for old prediction events
    asyncio.create_task(cleanup_prediction_events())

async def cleanup_prediction_events():
    """Remove stale events every 5 minutes"""
    while True:
        await asyncio.sleep(300)  # 5 mins
        
        stale_flows = [
            flow_id for flow_id in G.prediction_events.keys()
            if flow_id not in G.prediction_history
        ]
        
        for flow_id in stale_flows:
            G.prediction_events.pop(flow_id, None)
        
        if stale_flows:
            print(f"[CLEANUP] Removed {len(stale_flows)} stale events")

app.include_router(ws_router)
app.include_router(predict_router, prefix="/predict", tags=["Predict"])
app.include_router(feedback_router, prefix="/feedback", tags=["Feedback"])
app.include_router(metrics_router, tags=["Metrics"])

@app.get("/")
def root():
    return {
        "status": "running",
        "version": "9.0",
        "drift_detection": True,
        "training": False,
        "timeline_size":  len(G.drift_timeline)
    }
