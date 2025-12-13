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
    # FAST PATH: Check history first (O(1) lookup)
    # -----------------------
    prediction = G.prediction_history.get(data.flow_id)
    
    if prediction is None:
        # SLOW PATH: Wait for prediction
        # Tạo event nếu chưa có (trường hợp feedback đến trước predict)
        if data.flow_id not in G.prediction_events:
            G.prediction_events[data.flow_id] = asyncio.Event()
        
        try:
            # Đợi tối đa 5 giây (giảm từ 30s để tăng responsiveness)
            await asyncio.wait_for(
                G.prediction_events[data.flow_id].wait(), 
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] Flow {data.flow_id} not predicted after 5s")
            # Clean up event
            G.prediction_events.pop(data.flow_id, None)
            return {"error": "Unknown Flow ID", "reason": "timeout"}
        
        # Retry get after wait
        prediction = G.prediction_history.get(data.flow_id)
        
        # Double-check after wait
        if prediction is None:
            print(f"[ERROR] Flow {data.flow_id} still not in history after wait")
            G.prediction_events.pop(data.flow_id, None)
            return {"error": "Unknown Flow ID", "reason": "not_found"}
    
    # Unpack prediction (only one dict access)
    pred_label, pred_id = prediction
    
    # Clean up event sau khi dùng xong
    G.prediction_events.pop(data.flow_id, None)

    # -----------------------
    # LABEL ENCODING
    # -----------------------
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

            # Async DynamoDB update (non-blocking)
            asyncio.create_task(
                mark_drift_in_dynamodb(data.flow_id)
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

    # ----------------------------------------------------------
    #  NORMAL (NO DRIFT)
    # ----------------------------------------------------------
    # Batch metric updates (more efficient)
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


# ----------------------------------------------------------
#  HELPER: Async DynamoDB update (non-blocking)
# ----------------------------------------------------------
async def mark_drift_in_dynamodb(flow_id: str):
    """
    Update drift_detected flag in DynamoDB asynchronously.
    This prevents blocking the main feedback response.
    """
    try:
        # Query tất cả records với flow_id
        resp = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("flow_id").eq(flow_id)
        )

        # Update tất cả items
        for item in resp.get("Items", []):
            table.update_item(
                Key={
                    "flow_id": flow_id,
                    "timestamp": item["timestamp"]
                },
                UpdateExpression="SET drift_detected = :v",
                ExpressionAttributeValues={":v": True}
            )

        print(f"[DDB] Marked drift_detected=True for flow_id {flow_id}")
        
    except Exception as e:
        print(f"[DDB ERROR] Failed to mark drift for {flow_id}: {e}")