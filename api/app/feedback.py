from fastapi import APIRouter
from app.schemas import FeedbackSchema
import app.globals as G
from app.metrics import FEEDBACK_COUNT, LEARN_COUNT
from app.utils.preprocess import sanitize

router = APIRouter()

@router.post("")
def feedback(data: FeedbackSchema):
    FEEDBACK_COUNT.inc()

    print("\n====================== FEEDBACK ======================")
    print(f"Flow ID            = {data.flow_id}")
    print(f"True raw label     = '{data.true_label}'")
    print(f"prediction_history id = {id(G.prediction_history)}")

    # ------------------------------------------------------------
    # LABEL NORMALIZATION
    # ------------------------------------------------------------
    lookup = {c.lower(): c for c in G.encoder.classes_}
    true_raw = data.true_label.strip()

    if true_raw.lower() not in lookup:
        return {"error": f"Label not in encoder: {true_raw}"}

    normalized_label = lookup[true_raw.lower()]
    y_true = int(G.encoder.transform([normalized_label])[0])

    # ------------------------------------------------------------
    # PREDICTION HISTORY
    # ------------------------------------------------------------
    if data.flow_id not in G.prediction_history:
        return {"error": f"No prediction for Flow ID {data.flow_id}"}

    pred_str, pred_id = G.prediction_history[data.flow_id]

    is_error = int(pred_id != y_true)
    G.adwin.update(is_error)
    drift_detected = G.adwin.drift_detected

    # ------------------------------------------------------------
    # FEATURES + SCALER (FIXED)
    # ------------------------------------------------------------
    features = sanitize(data.features)

    # FIX 1 — update scaler
    G.scaler.learn_one(features)

    # FIX 2 — transform after learning scaler
    x = G.scaler.transform_one(features)

    with G.model_lock:

        # BEFORE LEARNING
        proba_before = G.model.predict_proba_one(x)
        pred_before = max(proba_before, key=proba_before.get)
        conf_before = float(proba_before[pred_before])

        need_learning = (pred_before != y_true) or (conf_before < 0.8)

        if need_learning:
            G.model.learn_one(x, y_true)
            LEARN_COUNT.inc()

        # AFTER LEARNING
        proba_after = G.model.predict_proba_one(x)
        pred_after = max(proba_after, key=proba_after.get)
        conf_after = float(proba_after[pred_after])

    return {
        "status": "ok",
        "flow_id": str(data.flow_id),
        "true_label": str(normalized_label),
        "pred_before": str(G.encoder.inverse_transform([int(pred_before)])[0]),
        "conf_before": round(conf_before, 4),
        "pred_after": str(G.encoder.inverse_transform([int(pred_after)])[0]),
        "conf_after": round(conf_after, 4),
        "need_learning": bool(need_learning),
        "drift_detected": bool(drift_detected),
    }
