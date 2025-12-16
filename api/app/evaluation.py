from fastapi import APIRouter, UploadFile, File
import pandas as pd
import io

from river import metrics
from app import globals as G

router = APIRouter()


def normalize_pred_label(y_pred, encoder):
    """
    Normalize model prediction to STRING label.
    Supports:
    - int
    - float / numpy scalar
    - string
    """
    # Case 1: model trả STRING (River chuẩn)
    if isinstance(y_pred, str):
        return y_pred.upper()

    # Case 2: model trả số (int / float)
    try:
        y_int = int(y_pred)        # 0.0 -> 0
        return encoder.inverse_transform([y_int])[0]
    except Exception:
        return None


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
    # 2. INIT METRICS (STRING-BASED)
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

        # ----- TRUE LABEL (STRING) -----
        y_true_label = str(row[label_col]).upper()

        # ----- FEATURES -----
        try:
            x_raw = row.drop(label_col).to_dict()
            x = {k: float(x_raw[k]) for k in G.FEATURE_ORDER}
        except Exception:
            continue

        x_scaled = G.scaler.transform_one(x)
        y_pred_raw = G.model.predict_one(x_scaled)

        if y_pred_raw is None:
            continue

        # ----- NORMALIZE PRED LABEL -----
        y_pred_label = normalize_pred_label(y_pred_raw, G.encoder)
        if y_pred_label is None:
            continue

        # ----- UPDATE METRICS (STRING–STRING) -----
        metric_acc.update(y_true_label, y_pred_label)
        metric_prec.update(y_true_label, y_pred_label)
        metric_rec.update(y_true_label, y_pred_label)
        metric_f1.update(y_true_label, y_pred_label)
        metric_kappa.update(y_true_label, y_pred_label)
        metric_cm.update(y_true_label, y_pred_label)

        # ----- DEBUG LOG (GIỮ ĐỂ SO SÁNH MODEL CŨ / MỚI) -----
        print(
            "[DEBUG]",
            "raw_pred =", y_pred_raw,
            "| type =", type(y_pred_raw),
            "| normalized =", y_pred_label
        )

        rows_used += 1

    # ======================================================
    # 4. FORMAT CONFUSION MATRIX
    # ======================================================
    labels = sorted(metric_cm.labels)

    cm = {t: {p: int(metric_cm[t][p]) for p in labels} for t in labels}

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
