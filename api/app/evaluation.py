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
    # 1. LOAD CSV
    # ======================================================
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Tìm cột label (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    if label_col is None:
        return {"error": "CSV must contain Label column"}

    # ======================================================
    # 2. INIT METRICS
    # ======================================================
    metric_acc = metrics.Accuracy()
    metric_prec = metrics.Precision()
    metric_rec = metrics.Recall()
    metric_f1 = metrics.F1()
    metric_kappa = metrics.CohenKappa()
    metric_cm = metrics.ConfusionMatrix()

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
            continue  # skip unknown label

        # ----- FEATURES -----
        try:
            x_raw = row.drop(label_col).to_dict()
            # đảm bảo đúng thứ tự feature đã train
            x = {k: float(x_raw[k]) for k in G.FEATURE_ORDER}
        except Exception:
            continue  # skip row lỗi feature

        x_scaled = G.scaler.transform_one(x)
        y_pred = G.model.predict_one(x_scaled)

        # ----- UPDATE METRICS -----
        metric_acc.update(y_true, y_pred)
        metric_prec.update(y_true, y_pred)
        metric_rec.update(y_true, y_pred)
        metric_f1.update(y_true, y_pred)
        metric_kappa.update(y_true, y_pred)
        metric_cm.update(y_true, y_pred)

        rows_used += 1

    # ======================================================
    # 4. FORMAT CONFUSION MATRIX
    # ======================================================
    # river ConfusionMatrix lưu dạng dict[(y_true, y_pred)] = count
    cm_raw = metric_cm.confusion_matrix

    labels = list(G.encoder.classes_)
    label_ids = list(range(len(labels)))

    cm = {
        labels[i]: {labels[j]: int(cm_raw.get((i, j), 0)) for j in label_ids}
        for i in label_ids
    }

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
