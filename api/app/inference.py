from fastapi import APIRouter
import time
from app.schemas import FlowSchema, PredictionResponse
from app.globals import model, scaler, encoder, model_lock, prediction_history
from app.metrics import PREDICTION_COUNT, LATENCY

router = APIRouter()


@router.post("", response_model=PredictionResponse)
def predict(flow: FlowSchema):
    start = time.time()

    print("\n=== [PREDICT] ======================")

    # Flow ID bắt buộc
    if flow.flow_id is None:
        print("ERROR: Missing Flow ID")
        return {"error": "Flow ID is required"}

    flow_id = flow.flow_id
    print("Flow ID:", flow_id)

    try:
        x = scaler.transform_one(flow.features)

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

        # Save history
        prediction_history[flow_id] = (decoded, int(pred))

        print(f"Saved prediction_history[{flow_id}] = ({decoded}, {int(pred)})")
        print("prediction_history id():", id(prediction_history))

        latency = (time.time() - start) * 1000
        LATENCY.observe(latency)
        PREDICTION_COUNT.inc()

        print(f"Prediction: {decoded} ({conf:.4f}), Latency={latency:.3f} ms")

        return PredictionResponse(
            flow_id=flow_id,
            prediction=str(decoded),
            confidence=round(conf, 4),
            latency_ms=round(latency, 3)
        )

    except Exception as e:
        print("[PREDICT ERROR]:", e)
        return {"error": str(e)}

#