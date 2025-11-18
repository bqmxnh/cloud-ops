from fastapi import APIRouter
from app.schemas import FeedbackSchema
from app.model_loader import model, scaler, encoder, model_lock
from app.metrics import FEEDBACK_COUNT, LEARN_COUNT
from app.utils.preprocess import sanitize

router = APIRouter()

@router.post("")
def feedback(data: FeedbackSchema):
    FEEDBACK_COUNT.inc()

    flow_id = data.flow_id
    true_label = data.true_label.strip().lower()

    try:
        y_true = int(encoder.transform([true_label])[0])
    except:
        return {"error": f"Invalid true_label: {data.true_label}"}

    features = sanitize(data.features)
    x = scaler.transform_one(features)

    with model_lock:

        # BEFORE
        proba_before = model.predict_proba_one(x)
        if proba_before:
            pred_before = max(proba_before, key=proba_before.get)
            conf_before = float(proba_before[pred_before])
        else:
            pred_before = model.predict_one(x)
            conf_before = 1.0

        need_learning = (pred_before != y_true) or (conf_before < 0.8)

        if need_learning:
            model.learn_one(x, y_true)
            LEARN_COUNT.inc()

        # AFTER
        proba_after = model.predict_proba_one(x)
        if proba_after:
            pred_after = max(proba_after, key=proba_after.get)
            conf_after = float(proba_after[pred_after])
        else:
            pred_after = model.predict_one(x)
            conf_after = 1.0

    try:
        decoded_before = encoder.inverse_transform([int(pred_before)])[0]
        decoded_after = encoder.inverse_transform([int(pred_after)])[0]
    except:
        decoded_before = pred_before
        decoded_after = pred_after

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
        "learned": need_learning
    }
