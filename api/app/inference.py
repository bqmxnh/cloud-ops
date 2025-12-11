from fastapi import APIRouter
from app.schemas import FlowSchema
from app import globals as G
from app.websocket import broadcast
import time

router = APIRouter()

@router.post("")
async def predict(flow: FlowSchema):

    start = time.time()

    if flow.flow_id is None:
        return {"error": "Flow ID required"}

    x_scaled = G.scaler.transform_one(flow.features)

    proba = G.model.predict_proba_one(x_scaled)
    if proba:
        pred = max(proba, key=proba.get)
        conf = float(proba[pred])
    else:
        pred = G.model.predict_one(x_scaled)
        conf = 1.0

    decoded = G.encoder.inverse_transform([int(pred)])[0]

    G.prediction_history[flow.flow_id] = (decoded, int(pred))

    # Websocket push
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
