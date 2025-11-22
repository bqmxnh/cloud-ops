from fastapi import APIRouter
from app.schemas import FeedbackSchema
from app.model_loader import (
    model, scaler, encoder, model_lock,
    prediction_history,
    metric_acc, metric_prec, metric_rec,
    metric_f1, metric_kappa,
    adwin
)
from app.metrics import FEEDBACK_COUNT, LEARN_COUNT
from app.utils.preprocess import sanitize

router = APIRouter()


@router.post("")
def feedback(data: FeedbackSchema):
    FEEDBACK_COUNT.inc()

    flow_id = data.flow_id
    true_label = data.true_label.strip().lower()

    # Encode true label
    try:
        y_true = int(encoder.transform([true_label])[0])
    except:
        return {"error": f"Invalid true_label: {data.true_label}"}

    # Match predict → feedback
    if flow_id not in prediction_history:
        return {"error": f"No prediction found for Flow ID {flow_id}"}

    (pred_str, pred_id) = prediction_history[flow_id]

    # DRIFT BEFORE LEARNING (NEW)
    is_error = int(pred_id != y_true)
    adwin.update(is_error)
    drift_detected = adwin.drift_detected

    # Preprocess data
    features = sanitize(data.features)
    x = scaler.transform_one(features)

    with model_lock:

        # BEFORE
        proba_before = model.predict_proba_one(x)
        pred_before = max(proba_before, key=proba_before.get)
        conf_before = float(proba_before[pred_before])

        need_learning = (pred_before != y_true) or (conf_before < 0.8)

        # INCREMENTAL LEARNING (same condition as 5.0)
        if need_learning:
            model.learn_one(x, y_true)
            LEARN_COUNT.inc()

        # AFTER
        proba_after = model.predict_proba_one(x)
        pred_after = max(proba_after, key=proba_after.get)
        conf_after = float(proba_after[pred_after])

    decoded_before = encoder.inverse_transform([int(pred_before)])[0]
    decoded_after = encoder.inverse_transform([int(pred_after)])[0]

    # GLOBAL METRICS UPDATE (NEW)
    metric_acc.update(y_true, pred_id)
    metric_prec.update(y_true, pred_id)
    metric_rec.update(y_true, pred_id)
    metric_f1.update(y_true, pred_id)
    metric_kappa.update(y_true, pred_id)

    return {
        "status": "ok",
        "flow_id": flow_id,
        "true_label": true_label,

        "pred_before": str(decoded_before),
        "conf_before": round(conf_before, 4),

        "pred_after": str(decoded_after),
        "conf_after": round(conf_after, 4),

        "delta_conf": round(conf_after - conf_before, 4),
        "need_learning": need_learning,
        "learned": need_learning,

        "drift_detected": drift_detected,

        "metrics": {
            "accuracy": float(metric_acc.get()),
            "precision": float(metric_prec.get()),
            "recall": float(metric_rec.get()),
            "f1": float(metric_f1.get()),
            "kappa": float(metric_kappa.get())
        }
    }
