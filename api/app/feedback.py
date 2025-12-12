from fastapi import APIRouter
from app.schemas import FeedbackSchema
from app import globals as G
from app.websocket import broadcast
from app.utils.drift_timeline import add_drift_event
from app.utils.preprocess import sanitize
import time
import boto3

router = APIRouter()

# ===== DynamoDB Table =====
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("ids_log_system")


@router.post("")
async def feedback(data: FeedbackSchema):

    import asyncio

    for _ in range(3):
        if data.flow_id in G.prediction_history:
            break
        await asyncio.sleep(0.008) 
    # -----------------------
    # VALIDATION
    # -----------------------
    if data.flow_id not in G.prediction_history:
        return {"error": "Unknown Flow ID"}

    pred_label, pred_id = G.prediction_history[data.flow_id]

    y_true = int(G.encoder.transform([data.true_label.upper()])[0])
    match = (pred_id == y_true)

    is_error = 0 if match else 1
    G.error_buffer.append(is_error)

    # -----------------------
    # ADWIN DRIFT DETECTOR
    # -----------------------
    G.adwin.update(is_error)
    drift_flag = G.adwin.drift_detected

    # ----------------------------------------------------------
    #  DRIFT CASE
    # ----------------------------------------------------------
    if drift_flag:
        recent_err = sum(G.error_buffer) / len(G.error_buffer)
        global_err = G.adwin.estimation

        is_degradation = recent_err > global_err

        if is_degradation:

            # Store drift event in local drift timeline (your existing logic)
            event = add_drift_event(recent_err, global_err, "degradation")

            # =======================================
            #  NEW: STORE DRIFT EVENT IN DYNAMODB
            # =======================================
            drift_ts = int(time.time() * 1000)

            table.put_item(Item={
                "pk": "drift_event",
                "timestamp": drift_ts,
                "flow_id": data.flow_id,
                "true_label": data.true_label,
                "predicted": pred_label,
                "reason": "performance_degradation",
                "type": "DRIFT"
            })

            print(f"[DYNAMODB] Logged DRIFT_EVENT at {drift_ts}")

            # =======================================
            #  SEND WEBSOCKET NOTIFICATION
            # =======================================
            await broadcast("drift_event", {
                "timestamp": drift_ts,
                "flow_id": data.flow_id,
                "recent_err": recent_err,
                "global_err": global_err,
                "type": "DRIFT"
            })

            # =======================================
            #  RESPONSE TO CLIENT/UI
            # =======================================
            return {
                "flow_id": data.flow_id,
                "predicted": pred_label,
                "true_label": data.true_label,
                "match": match,
                "drift_detected": True,
                "timestamp": drift_ts,
                "reason": "performance degradation"
            }

    # ----------------------------------------------------------
    #  NORMAL (NO DRIFT)
    # ----------------------------------------------------------
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
