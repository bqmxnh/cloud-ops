from fastapi import APIRouter
import time
from app.schemas import FlowSchema, PredictionResponse
from app.model_loader import model, scaler, encoder, model_lock, prediction_history
from app.metrics import PREDICTION_COUNT, LATENCY

router = APIRouter()


@router.post("", response_model=PredictionResponse)
def predict(flow: FlowSchema):
    start = time.time()

    # === 1) Giống bản cũ: Flow ID bắt buộc ===
    if flow.flow_id is None:
        return {"error": "Flow ID is required"}

    flow_id = flow.flow_id  # không fallback, bắt buộc đúng

    try:
        # === 2) Giống bản cũ: không sanitize ===
        x = scaler.transform_one(flow.features)

        with model_lock:
            proba = model.predict_proba_one(x)

            if proba:
                pred = max(proba, key=proba.get)
                conf = float(proba[pred])
            else:
                pred = model.predict_one(x)
                conf = 1.0

            # decode giống bản cũ
            try:
                decoded = encoder.inverse_transform([int(pred)])[0]
            except:
                decoded = pred

        # === 3) Giống bản cũ: lưu (decoded_string, int_pred) ===
        prediction_history[flow_id] = (decoded, int(pred))

        # === 4) Metrics ===
        latency = (time.time() - start) * 1000
        LATENCY.observe(latency)
        PREDICTION_COUNT.inc()

        return PredictionResponse(
            flow_id=flow_id,
            prediction=str(decoded),
            confidence=round(conf, 4),
            latency_ms=round(latency, 3)
        )

    except Exception as e:
        return {"error": str(e)}
