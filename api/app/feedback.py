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
    true_raw = data.true_label.strip()

    # ============================================================
    # 1) FIX LABEL ENCODER – luôn match đúng class gốc của encoder
    # ============================================================
    classes = list(encoder.classes_)          # ví dụ: ["BENIGN", "ATTACK"]
    lookup = {c.lower(): c for c in classes}  # {"benign":"BENIGN", "attack":"ATTACK"}

    if true_raw.lower() not in lookup:
        return {"error": f"Label not in encoder: {true_raw}"}

    normalized_label = lookup[true_raw.lower()]   # chuẩn hóa về "ATTACK"
    y_true = int(encoder.transform([normalized_label])[0])

    # ============================================================
    # 2) FLOW ID phải tồn tại từ bước predict
    # ============================================================
    if flow_id not in prediction_history:
        return {"error": f"No prediction found for Flow ID {flow_id}"}

    # pred_str = decoded label, pred_id = model raw int prediction
    (pred_str, pred_id) = prediction_history[flow_id]

    # ============================================================
    # 3) DRIFT BEFORE LEARNING
    # ============================================================
    is_error = int(pred_id != y_true)     # 1 nếu dự đoán sai → ADWIN xem đây là error
    adwin.update(is_error)
    drift_detected = adwin.drift_detected

    # ============================================================
    # 4) Preprocess features
    # ============================================================
    features = sanitize(data.features)
    x = scaler.transform_one(features)

    # ============================================================
    # 5) BEFORE + LEARN + AFTER
    # ============================================================
    with model_lock:

        # BEFORE
        proba_before = model.predict_proba_one(x)
        pred_before = max(proba_before, key=proba_before.get)
        conf_before = float(proba_before[pred_before])

        need_learning = (pred_before != y_true) or (conf_before < 0.8)

        # LEARN (incremental)
        if need_learning:
            model.learn_one(x, y_true)
            LEARN_COUNT.inc()

        # AFTER
        proba_after = model.predict_proba_one(x)
        pred_after = max(proba_after, key=proba_after.get)
        conf_after = float(proba_after[pred_after])

    # Decode labels
    decoded_before = encoder.inverse_transform([int(pred_before)])[0]
    decoded_after = encoder.inverse_transform([int(pred_after)])[0]

    # ============================================================
    # 6) UPDATE GLOBAL METRICS
    # ============================================================
    metric_acc.update(y_true, pred_id)
    metric_prec.update(y_true, pred_id)
    metric_rec.update(y_true, pred_id)
    metric_f1.update(y_true, pred_id)
    metric_kappa.update(y_true, pred_id)

    # ============================================================
    # 7) RETURN RESPONSE
    # ============================================================
    return {
        "status": "ok",
        "flow_id": flow_id,
        "true_label": normalized_label,

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
