from fastapi import APIRouter
import time
from app.schemas import FlowSchema, PredictionResponse
from app.model_loader import model, scaler, encoder, model_lock
from app.metrics import PREDICTION_COUNT, LATENCY
from app.utils.preprocess import sanitize

router = APIRouter()

@router.post("", response_model=PredictionResponse)
def predict(flow: FlowSchema):
    start = time.time()

    try:
        features = sanitize(flow.features)
        x = scaler.transform_one(features)

        with model_lock:
            proba = model.predict_proba_one(x)

            if proba:
                pred = max(proba, key=proba.get)
                conf = float(proba[pred])
            else:
                pred = model.predict_one(x)
                conf = 1.0

            try:
                decoded = encoder.inverse_transform([int(pred)])[0]
            except:
                decoded = pred

        latency = (time.time() - start) * 1000
        LATENCY.observe(latency)
        PREDICTION_COUNT.inc()

        return PredictionResponse(
            flow_id=flow.flow_id or "unknown",
            prediction=str(decoded),
            confidence=round(conf, 4),
            latency_ms=round(latency, 3)
        )

    except Exception as e:
        return {"error": str(e)}
