from fastapi import APIRouter, UploadFile, File
import pandas as pd
import io

from river import metrics
from app import globals as G

router = APIRouter()


@router.post("")
async def evaluate(file: UploadFile = File(...)):
    """
    Shadow / Online Evaluation using current deployed model
    CSV format: CIC-DDoS style, must contain Label column
    """

    # ======================================================
    # 0. SAFETY CHECK
    # ======================================================
    if G.model is None or G.scaler is None or G.encoder is None:
        return {"error": "Model not loaded"}

    # ======================================================
    # 1. LOAD CSV
    # ======================================================
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        return {"error": "CSV must contain Label column"}

    # ======================================================
    # 2. INIT METRICS
    # ======================================================
    metric_acc    = metrics.Accuracy()
    metric_prec   = metrics.Precision()
    metric_rec    = metrics.Recall()
    metric_f1     = metrics.F1()
    metric_kappa  = metrics.CohenKappa()
    metric_cm     = metrics.ConfusionMatrix()

    rows_used = 0

    # ======================================================
    # 3. EVALUATION LOOP (ONLINE STYLE)
    # ======================================================
    for _, row in df.iterrows():

        # ----- TRUE LABEL -----
        y_true_label = str(row[label_col]).upper()
        try:
            y_true = int(G.encoder.transform([y_true_label])[0])
        except Exception:
            continue

        # ----- FEATURES -----
        try:
            x_raw = row.drop(label_col).to_dict()
            x = {k: float(x_raw[k]) for k in G.FEATURE_ORDER}
        except Exception:
            continue

        x_scaled = G.scaler.transform_one(x)
        y_pred = G.model.predict_one(x_scaled)

        if y_pred is None:
            continue

        try:
            y_pred = int(y_pred)
        except Exception:
            continue

        # ----- UPDATE METRICS -----
        metric_acc.update(y_true, y_pred)
        metric_prec.update(y_true, y_pred)
        metric_rec.update(y_true, y_pred)
        metric_f1.update(y_true, y_pred)
        metric_kappa.update(y_true, y_pred)
        print(
            "[DEBUG]",
            "type(y_pred) =", type(y_pred),
            "| value =", y_pred
        )


        y_true_str = G.encoder.inverse_transform([y_true])[0]
        y_pred_str = G.encoder.inverse_transform([y_pred])[0]
        metric_cm.update(y_true_str, y_pred_str)

        rows_used += 1

    # ======================================================
    # 4. FORMAT CONFUSION MATRIX (RIVER OFFICIAL API)
    # ======================================================
    labels = list(G.encoder.classes_)

    cm = {t: {p: 0 for p in labels} for t in labels}

    for t in labels:
        for p in labels:
            try:
                cm[t][p] = int(metric_cm[t][p])
            except KeyError:
                cm[t][p] = 0

    # ======================================================
    # 5. RETURN RESULT
    # ======================================================
    return {
        "rows_evaluated": rows_used,
        "accuracy": round(metric_acc.get(), 6),
        "precision": round(metric_prec.get(), 6),
        "recall": round(metric_rec.get(), 6),
        "f1": round(metric_f1.get(), 6),
        "kappa": round(metric_kappa.get(), 6),
        "confusion_matrix": cm
    }
