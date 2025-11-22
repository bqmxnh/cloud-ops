from fastapi import APIRouter
from app.schemas import FeedbackSchema
import app.globals as G
from app.metrics import FEEDBACK_COUNT, LEARN_COUNT
from app.utils.preprocess import sanitize

router = APIRouter()

@router.post("")
def feedback(data: FeedbackSchema):
    FEEDBACK_COUNT.inc()

    print("\n=== [FEEDBACK] ======================")

    flow_id = data.flow_id
    true_raw = data.true_label.strip()

    print("Flow ID:", flow_id)
    print("True raw:", true_raw)
    print("prediction_history id:", id(G.prediction_history))

    # ----- MANUAL LABEL MAP (bỏ encoder mapping bị sai) -----
    manual_map = {"benign": 0, "attack": 1}

    if true_raw.lower() not in manual_map:
        return {"error": f"Invalid true_label: {true_raw}"}

    y_true = manual_map[true_raw.lower()]
    normalized_label = true_raw.upper()
    # -----------------------------------------------------------

    if flow_id not in G.prediction_history:
        return {"error": f"No prediction for Flow ID {flow_id}"}

    pred_str, pred_id = G.prediction_history[flow_id]

    # ===== DRIFT BEFORE LEARNING =====
    is_error = int(pred_id != y_true)
    G.adwin.update(is_error)
    drift_detected = G.adwin.drift_detected

    features = sanitize(data.features)
    x = G.scaler.transform_one(features)

    with G.model_lock:
        # BEFORE
        proba_before = G.model.predict_proba_one(x)
        pred_before = max(proba_before, key=proba_before.get)
        conf_before = float(proba_before[pred_before])

        need_learning = (pred_before != y_true) or (conf_before < 0.8)
        if need_learning:
            G.model.learn_one(x, y_true)
            LEARN_COUNT.inc()

        # AFTER
        proba_after = G.model.predict_proba_one(x)
        pred_after = max(proba_after, key=proba_after.get)
        conf_after = float(proba_after[pred_after])

    # ---- MANUAL INVERSE MAP ----
    inv_map = {0: "BENIGN", 1: "ATTACK"}
    decoded_before = inv_map.get(int(pred_before), "UNKNOWN")
    decoded_after  = inv_map.get(int(pred_after), "UNKNOWN")
    # ------------------------------

    # ===== METRICS =====
    G.metric_acc.update(y_true, pred_id)
    G.metric_prec.update(y_true, pred_id)
    G.metric_rec.update(y_true, pred_id)
    G.metric_f1.update(y_true, pred_id)
    G.metric_kappa.update(y_true, pred_id)

    return {
        "status": "ok",
        "flow_id": flow_id,
        "true_label": normalized_label,
        "pred_before": decoded_before,
        "conf_before": round(conf_before, 4),
        "pred_after": decoded_after,
        "conf_after": round(conf_after, 4),
        "need_learning": need_learning,
        "drift_detected": drift_detected
    }
