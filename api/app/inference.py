# app/routers/predict.py
from fastapi import APIRouter
from app.schemas import FlowSchema
from app import globals as G
from app.websocket import broadcast
from app.metrics import PRED_COUNT, LATENCY
import time
import asyncio
import math
from collections import defaultdict


router = APIRouter()

def entropy(p: dict) -> float:
    return -sum(v * math.log(v + 1e-9) for v in p.values())

def predict_proba_confident(model, x, entropy_th=0.25, min_trees=3):
    """
    Aggregate probabilities only from confident trees (low entropy).
    Fallback to full ensemble if not enough trees are confident.
    """
    # If the model is not an ensemble with `models` (e.g., KNN, HAT),
    # just return its probability prediction directly.
    if not hasattr(model, "models") or model.models is None:
        try:
            return model.predict_proba_one(x) or {}
        except Exception:
            return {}

    agg = defaultdict(float)
    used = 0

    # ARFClassifier keeps base models in `models`
    try:
        for tree in model.models:
            p = tree.predict_proba_one(x)
            if not p:
                continue

            if entropy(p) < entropy_th:
                for k, v in p.items():
                    agg[k] += v
                used += 1
    except Exception:
        # If anything goes wrong, fall back to model-level probability
        try:
            return model.predict_proba_one(x) or {}
        except Exception:
            return {}

    # Fallback if not enough confident trees
    if used < min_trees:
        try:
            return model.predict_proba_one(x) or {}
        except Exception:
            return {}

    s = sum(agg.values())
    if s == 0:
        # Avoid division by zero; fall back to model-level probability
        try:
            return model.predict_proba_one(x) or {}
        except Exception:
            return {}

    return {k: v / s for k, v in agg.items()}


@router.post("")
async def predict(flow: FlowSchema):
    start = time.time()

    PRED_COUNT.inc() 

    if flow.flow_id is None:
        return {"error": "Flow ID required"}

    if flow.flow_id not in G.prediction_events:
        G.prediction_events[flow.flow_id] = asyncio.Event()

    x_scaled = G.scaler.transform_one(flow.features)

    proba = predict_proba_confident(
        G.model,
        x_scaled,
        entropy_th=0.25,
        min_trees=3
    )

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
