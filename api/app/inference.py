# app/routers/predict.py
from fastapi import APIRouter
from app.schemas import FlowSchema
from app import globals as G
from app.websocket import broadcast
from app.metrics import PRED_COUNT, LATENCY
import time
import asyncio

router = APIRouter()

@router.post("")
async def predict(flow: FlowSchema):
    start = time.time()

    PRED_COUNT.inc() 

    if flow.flow_id is None:
        return {"error": "Flow ID required"}

    if flow.flow_id not in G.prediction_events:
        G.prediction_events[flow.flow_id] = asyncio.Event()

    x_scaled = G.scaler.transform_one(flow.features)
    proba = G.model.predict_proba_one(x_scaled)

    if proba:
        pred = max(proba, key=proba.get)
        conf = float(proba[pred])
    else:
        pred = G.model.predict_one(x_scaled)
        conf = 1.0

    decoded = G.encoder.inverse_transform([int(pred)])[0]

    G.prediction_events[flow.flow_id].set()

    await broadcast("new_flow", {
        "flow_id": flow.flow_id,
        "prediction": decoded,
        "confidence": conf,
        "features": flow.features
    })

    latency = (time.time() - start) * 1000
    LATENCY.observe(latency)  

    return {
        "flow_id": flow.flow_id,
        "prediction": decoded,
        "confidence": conf,
        "latency_ms": round(latency, 3)
    }
