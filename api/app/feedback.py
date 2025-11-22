from fastapi import APIRouter
from app.schemas import FeedbackSchema
import app.globals as G
from app.metrics import FEEDBACK_COUNT, LEARN_COUNT
from app.utils.preprocess import sanitize

router = APIRouter()

@router.post("")
def feedback(data: FeedbackSchema):
    FEEDBACK_COUNT.inc()

    print("\n====================== FEEDBACK ======================")
    print(f"Flow ID            = {data.flow_id}")
    print(f"True raw label     = '{data.true_label}'")
    print(f"prediction_history id = {id(G.prediction_history)}")

    # ============================================================
    # 1️⃣ LABEL NORMALIZATION DEBUG
    # ============================================================
    print("\n--- LABEL ENCODER DEBUG ---")
    print(f"Classes in encoder = {list(G.encoder.classes_)}")

    lookup = {c.lower(): c for c in G.encoder.classes_}
    true_raw = data.true_label.strip()

    print(f"Lookup table       = {lookup}")

    if true_raw.lower() not in lookup:
        print("❌ ERROR: LABEL NOT FOUND IN LOOKUP")
        return {"error": f"Label not in encoder: {true_raw}"}

    normalized_label = lookup[true_raw.lower()]
    y_true = int(G.encoder.transform([normalized_label])[0])

    print(f"Normalized label   = {normalized_label}")
    print(f"Encoded y_true     = {y_true}")

    # ============================================================
    # 2️⃣ PREDICTION HISTORY DEBUG
    # ============================================================
    if data.flow_id not in G.prediction_history:
        print("❌ ERROR: FLOW ID NOT FOUND IN prediction_history")
        return {"error": f"No prediction for Flow ID {data.flow_id}"}

    pred_str, pred_id = G.prediction_history[data.flow_id]

    print("\n--- PREDICTION HISTORY DEBUG ---")
    print(f"pred_str (decoded) = {pred_str}")
    print(f"pred_id (encoded)  = {pred_id}")
    print(f"y_true             = {y_true}")
    print(f"is_error           = {int(pred_id != y_true)} (1=wrong,0=correct)")

    # Drift detector
    G.adwin.update(int(pred_id != y_true))
    drift_detected = G.adwin.drift_detected
    print(f"Drift detected     = {drift_detected}")

    # ============================================================
    # 3️⃣ FEATURE + SCALER DEBUG
    # ============================================================
    features = sanitize(data.features)
    print("\n--- FEATURES DEBUG ---")
    print("Raw feature sample =", list(features.items())[:5])

    x = G.scaler.transform_one(features)
    print("Scaled feature sample =", list(x.items())[:5])

    # ============================================================
    # 4️⃣ MODEL PREDICTION BEFORE LEARNING
    # ============================================================
    with G.model_lock:
        proba_before = G.model.predict_proba_one(x)
        pred_before = max(proba_before, key=proba_before.get)
        conf_before = float(proba_before[pred_before])

        print("\n--- BEFORE LEARNING ---")
        print(f"pred_before        = {pred_before}")
        print(f"decoded_before     = {G.encoder.inverse_transform([int(pred_before)])[0]}")
        print(f"conf_before        = {conf_before}")

        # ========================================================
        # 5️⃣ LEARNING DECISION DEBUG
        # ========================================================
        need_learning = (pred_before != y_true) or (conf_before < 0.8)
        print(f"\n--- LEARNING DECISION ---")
        print(f"pred_before != y_true ?  {pred_before != y_true}")
        print(f"conf_before < 0.8  ?    {conf_before < 0.8}")
        print(f"=> need_learning   =    {need_learning}")

        if need_learning:
            print("✔ APPLY LEARNING NOW: model.learn_one(x, y_true)")
            G.model.learn_one(x, y_true)
            LEARN_COUNT.inc()
        else:
            print("✖ NO LEARNING APPLIED")

        # ========================================================
        # 6️⃣ AFTER LEARNING DEBUG
        # ========================================================
        proba_after = G.model.predict_proba_one(x)
        pred_after = max(proba_after, key=proba_after.get)
        conf_after = float(proba_after[pred_after])

        print("\n--- AFTER LEARNING ---")
        print(f"pred_after         = {pred_after}")
        print(f"decoded_after      = {G.encoder.inverse_transform([int(pred_after)])[0]}")
        print(f"conf_after         = {conf_after}")

    print("====================== END FEEDBACK ======================\n")

    return {
        "status": "ok",
        "flow_id": data.flow_id,
        "true_label": normalized_label,
        "pred_before": G.encoder.inverse_transform([int(pred_before)])[0],
        "conf_before": round(conf_before, 4),
        "pred_after": G.encoder.inverse_transform([int(pred_after)])[0],
        "conf_after": round(conf_after, 4),
        "need_learning": need_learning,
        "drift_detected": drift_detected
    }
