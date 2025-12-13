# app/routers/feedback.py
from fastapi import APIRouter
from app.schemas import FeedbackSchema
from app import globals as G
from app.websocket import broadcast
from app.utils.drift_timeline import add_drift_event
import time
import boto3
from boto3.dynamodb.conditions import Key

router = APIRouter()

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("ids_log_system")


@router.post("")
async def feedback(data: FeedbackSchema):

    # ======================================================
    # 1. LẤY PREDICTION TỪ DYNAMODB (SOURCE OF TRUTH)
    # ======================================================
    resp = table.query(
        KeyConditionExpression=Key("flow_id").eq(data.flow_id),
        ScanIndexForward=False,   # newest first
        Limit=1
    )

    items = resp.get("Items", [])
    if not items:
        return {
            "error": "Unknown Flow ID",
            "reason": "not_found_in_db"
        }

    item = items[0]

    pred_label = item.get("label")          # benign / attack
    pred_label = pred_label.upper()
    pred_id = int(G.encoder.transform([pred_label])[0])

    # ======================================================
    # 2. UPDATE TRUE LABEL VÀO DB
    # ======================================================
    table.update_item(
        Key={
            "flow_id": data.flow_id,
            "timestamp": item["timestamp"]
        },
        UpdateExpression="SET true_label = :v",
        ExpressionAttributeValues={
            ":v": data.true_label.lower()
        }
    )

    # ======================================================
    # 3. SO KHỚP LABEL → ERROR SIGNAL
    # ======================================================
    y_true = int(G.encoder.transform([data.true_label.upper()])[0])
    match = (pred_id == y_true)

    is_error = 0 if match else 1
    G.error_buffer.append(is_error)

    # ======================================================
    # 4. ADWIN DRIFT DETECTION
    # ======================================================
    G.adwin.update(is_error)
    drift_flag = G.adwin.drift_detected

    # ======================================================
    # 5. DRIFT CASE
    # ======================================================
    if drift_flag:
        recent_err = sum(G.error_buffer) / len(G.error_buffer)
        global_err = G.adwin.estimation

        if recent_err > global_err * 1.1:
            drift_ts = int(time.time() * 1000)

            # Đánh dấu drift trong DB
            table.update_item(
                Key={
                    "flow_id": data.flow_id,
                    "timestamp": item["timestamp"]
                },
                UpdateExpression="SET drift_detected = :v",
                ExpressionAttributeValues={":v": True}
            )

            await broadcast("drift_event", {
                "timestamp": drift_ts,
                "flow_id": data.flow_id,
                "recent_err": recent_err,
                "global_err": global_err,
                "type": "DRIFT"
            })

            return {
                "flow_id": data.flow_id,
                "predicted": pred_label,
                "true_label": data.true_label,
                "match": match,
                "drift_detected": True,
                "timestamp": drift_ts,
                "reason": "performance_degradation"
            }

    # ======================================================
    # 6. NORMAL METRICS UPDATE
    # ======================================================
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
