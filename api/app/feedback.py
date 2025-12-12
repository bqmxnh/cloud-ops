# app/routers/feedback.py
from fastapi import APIRouter
from app.schemas import FeedbackSchema
from app import globals as G
from app.websocket import broadcast
from app.utils.drift_timeline import add_drift_event
import time
import boto3
import asyncio

router = APIRouter()

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("ids_log_system")


@router.post("")
async def feedback(data: FeedbackSchema):

    # -----------------------
    # WAIT FOR PREDICTION
    # -----------------------
    if data.flow_id not in G.prediction_history:
        # Tạo event nếu chưa có (trường hợp feedback đến trước predict)
        if data.flow_id not in G.prediction_events:
            G.prediction_events[data.flow_id] = asyncio.Event()
        
        try:
            # Đợi tối đa 10 giây
            await asyncio.wait_for(
                G.prediction_events[data.flow_id].wait(), 
                timeout=10.0
            )
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] Flow {data.flow_id} not predicted after 10s")
            return {"error": "Unknown Flow ID", "reason": "timeout"}

    # -----------------------
    # VALIDATION
    # -----------------------
    if data.flow_id not in G.prediction_history:
        print(f"[ERROR] Flow {data.flow_id} still not in history after wait")
        return {"error": "Unknown Flow ID", "reason": "not_found"}

    pred_label, pred_id = G.prediction_history[data.flow_id]

    # Clean up event sau khi dùng xong
    G.prediction_events.pop(data.flow_id, None)

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

        is_degradation = recent_err > global_err * 1.1  # 10% worse

        if is_degradation:
            event = add_drift_event(recent_err, global_err, "degradation")
            drift_ts = int(time.time() * 1000)

            # Query DynamoDB
            resp = table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key("flow_id").eq(data.flow_id)
            )

            for item in resp.get("Items", []):
                table.update_item(
                    Key={
                        "flow_id": data.flow_id,
                        "timestamp": item["timestamp"]
                    },
                    UpdateExpression="SET drift_detected = :v",
                    ExpressionAttributeValues={":v": True}
                )

            print(f"[DDB] Marked drift_detected=True for flow_id {data.flow_id}")

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