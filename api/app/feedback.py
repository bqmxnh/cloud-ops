from fastapi import APIRouter
from app.schemas import FeedbackSchema
from app import globals as G
from app.websocket import broadcast
from app.utils.drift_timeline import add_drift_event
from app.utils.preprocess import sanitize

router = APIRouter()

@router.post("")
async def feedback(data: FeedbackSchema):

    if data.flow_id not in G.prediction_history:
        return {"error": "Unknown Flow ID"}

    pred_label, pred_id = G.prediction_history[data.flow_id]

    y_true = int(G.encoder.transform([data.true_label.upper()])[0])
    match = (pred_id == y_true)

    is_error = 0 if match else 1
    G.error_buffer.append(is_error)

    # ADWIN
    G.adwin.update(is_error)
    drift_flag = G.adwin.drift_detected

    # DRIFT EVALUATION
    if drift_flag:
        recent_err = sum(G.error_buffer) / len(G.error_buffer)
        global_err = G.adwin.estimation

        is_degradation = recent_err > global_err

        if is_degradation:
            event = add_drift_event(recent_err, global_err, "degradation")

            await broadcast("drift_event", event)

            return {
                "flow_id": data.flow_id,
                "predicted": pred_label,
                "true_label": data.true_label,
                "match": match,
                "drift_detected": True,
                "reason": "performance degradation"
            }

    # NO DRIFT → normal metrics
    G.metric_acc.update(y_true, pred_id)
    G.metric_prec.update(y_true, pred_id)
    G.metric_rec.update(y_true, pred_id)
    G.metric_f1.update(y_true, pred_id)
    G.metric_kappa.update(y_true, pred_id)
    G.metric_cm.update(y_true, pred_id)

    return {
        "flow_id": data.flow_id,
        "predicted": pred_label,
        "true_label": data.true_label,
        "match": match,
        "drift_detected": False
    }
