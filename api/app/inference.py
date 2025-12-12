# app/routers/predict.py
from fastapi import APIRouter
from app.schemas import FlowSchema
from app import globals as G
from app.websocket import broadcast
import time
import asyncio

router = APIRouter()

@router.post("")
async def predict(flow: FlowSchema):
    start = time.time()

    if flow.flow_id is None:
        return {"error": "Flow ID required"}

    # Create event TRƯỚC KHI predict (để feedback có thể đợi)
    if flow.flow_id not in G.prediction_events:
        G.prediction_events[flow.flow_id] = asyncio.Event()

    # Prediction logic
    x_scaled = G.scaler.transform_one(flow.features)
    proba = G.model.predict_proba_one(x_scaled)
    
    if proba:
        pred = max(proba, key=proba.get)
        conf = float(proba[pred])
    else:
        pred = G.model.predict_one(x_scaled)
        conf = 1.0

    decoded = G.encoder.inverse_transform([int(pred)])[0]

    # ATOMIC: Store prediction + Notify waiting feedbacks
    G.prediction_history[flow.flow_id] = (decoded, int(pred))
    G.prediction_events[flow.flow_id].set()  # 🔔 Signal đã sẵn sàng!

    # Websocket broadcast
    await broadcast("new_flow", {
        "flow_id": flow.flow_id,
        "prediction": decoded,
        "confidence": conf,
        "features": flow.features
    })

    latency = (time.time() - start) * 1000

    return {
        "flow_id": flow.flow_id,
        "prediction": decoded,
        "confidence": conf,
        "latency_ms": round(latency, 3)
    }