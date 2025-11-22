# app/routers/feedback.py

from fastapi import APIRouter
from app.schemas import FeedbackSchema
from app.globals import (
    model, scaler, encoder, model_lock,
    prediction_history,
    metric_acc, metric_prec, metric_rec,
    metric_f1, metric_kappa, adwin
)
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
    print("True raw label:", true_raw)
    print("prediction_history id():", id(prediction_history))

    # ===== normalize label =====
    classes = list(encoder.classes_)
    lookup = {c.lower(): c for c in classes}

    if true_raw.lower() not in lookup:
        print("ERROR: Unknown label")
        return {"error": f"Label not in encoder: {true_raw}"}

    normalized_label = lookup[true_raw.lower()]
    y_true = int(encoder.transform([normalized_label])[0])
    print("Normalized:", normalized_label, "| y_true:", y_true)

    # ===== check prediction exists =====
    if flow_id not in prediction_history:
        print("ERROR: Flow ID not found in prediction_history")
        return {"error": f"No prediction found for Flow ID {flow_id}"}

    pred_str, pred_id = prediction_history[flow_id]
    print("History:", prediction_history[flow_id])

    # ===== drift before learning =====
    is_error = int(pred_id != y_true)
    adwin.update(is_error)
    drift_detected = adwin.drift_detected

    print("pred_id:", pred_id, "| y_true:", y_true, "| is_error:", is_error)
    print("DRIFT:", drift_detected)

    # ===== preprocess + x =====
    features = sanitize(data.features)
    x = scaler.transform_one(features)

    with model_lock:

        proba_before = model.predict_proba_one(x)
        pred_before = max(proba_before, key=proba_before.get)
        conf_before = float(proba_before[pred_before])

        print(f"BEFORE: pred={pred_before}, conf={conf_before:.4f}")

        need_learning = (pred_before != y_true) or (conf_before < 0.8)
        print("need_learning:", need_learning)

        if need_learning:
            model.learn_one(x, y_true)
            LEARN_COUNT.inc()
            print("→ LEARNING APPLIED")

        proba_after = model.predict_proba_one(x)
        pred_after = max(proba_after, key=proba_after.get)
        conf_after = float(proba_after[pred_after])

        print(f"AFTER: pred={pred_after}, conf={conf_after:.4f}")

    decoded_before = encoder.inverse_transform([int(pred_before)])[0]
    decoded_after = encoder.inverse_transform([int(pred_after)])[0]

    # ===== update metrics =====
    metric_acc.update(y_true, pred_id)
    metric_prec.update(y_true, pred_id)
    metric_rec.update(y_true, pred_id)
    metric_f1.update(y_true, pred_id)
    metric_kappa.update(y_true, pred_id)

    return {
        "status": "ok",
        "flow_id": flow_id,
        "true_label": normalized_label,

        "pred_before": str(decoded_before),
        "conf_before": round(conf_before, 4),

        "pred_after": str(decoded_after),
        "conf_after": round(conf_after, 4),

        "need_learning": need_learning,
        "drift_detected": drift_detected,
    }
