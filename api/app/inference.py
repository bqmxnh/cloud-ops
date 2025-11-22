from fastapi import APIRouter
import time
from app.schemas import FlowSchema, PredictionResponse
import app.globals as G
from app.metrics import PREDICTION_COUNT, LATENCY

router = APIRouter()

@router.post("", response_model=PredictionResponse)
def predict(flow: FlowSchema):
    start = time.time()

    print("\n=== [PREDICT] ======================")

    if flow.flow_id is None:
        print("ERROR: Missing Flow ID")
        return {"error": "Flow ID is required"}

    flow_id = flow.flow_id
    print("Flow ID:", flow_id)

    try:
        x = G.scaler.transform_one(flow.features)

        with G.model_lock:
            proba = G.model.predict_proba_one(x)

            if proba:
                pred = max(proba, key=proba.get)
                conf = float(proba[pred])
            else:
                pred = G.model.predict_one(x)
                conf = 1.0

        # ---- MANUAL INVERSE MAP (đúng chuẩn BENIGN=0, ATTACK=1) ----
        inv_map = {0: "BENIGN", 1: "ATTACK"}
        decoded = inv_map.get(int(pred), "UNKNOWN")

        # save history
        G.prediction_history[flow_id] = (decoded, int(pred))

        print(f"Saved prediction_history[{flow_id}] = ({decoded}, {int(pred)})")
        print("prediction_history id():", id(G.prediction_history))

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
        print("[PREDICT ERROR]:", e)
        return {"error": str(e)}
