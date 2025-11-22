import os, joblib, tempfile, threading, boto3
from river import metrics, drift

BUCKET = os.getenv("MODEL_BUCKET", "arf-ids-model-bucket")
VERSION = os.getenv("MODEL_VERSION", "v1.0")


# ============================================================
# Load model from S3
# ============================================================
def load_from_s3(name: str):
    s3 = boto3.client("s3")
    key = f"{VERSION}/{name}"
    print(f"Downloading s3://{BUCKET}/{key}")

    with tempfile.NamedTemporaryFile() as tmp:
        s3.download_file(BUCKET, key, tmp.name)
        return joblib.load(tmp.name)


try:
    model = load_from_s3("model.pkl")
    scaler = load_from_s3("scaler.pkl")
    encoder = load_from_s3("label_encoder.pkl")
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed loading model: {e}")
    raise


# ============================================================
# Thread lock for safe incremental learning
# ============================================================
model_lock = threading.Lock()


# ============================================================
# GLOBAL METRICS FOR EVALUATION (NEW)
# ============================================================
metric_acc = metrics.Accuracy()
metric_prec = metrics.Precision()
metric_rec = metrics.Recall()
metric_f1 = metrics.F1()
metric_kappa = metrics.CohenKappa()


# ============================================================
# GLOBAL DRIFT DETECTOR (NEW)
# ============================================================
adwin = drift.ADWIN(delta=0.01)


# ============================================================
# GLOBAL MAP: predict → feedback (NEW)
# ============================================================
prediction_history = {}  # flow_id → (decoded_label, encoded_label)
