import os, joblib, tempfile, boto3
from app import globals as G   # dùng chung state

BUCKET = os.getenv("MODEL_BUCKET", "arf-ids-model-bucket")
VERSION = os.getenv("MODEL_VERSION", "v1.0")


def load_from_s3(name: str):
    s3 = boto3.client("s3")
    key = f"{VERSION}/{name}"
    print(f"[MODEL] Downloading s3://{BUCKET}/{key}")

    with tempfile.NamedTemporaryFile() as tmp:
        s3.download_file(BUCKET, key, tmp.name)
        return joblib.load(tmp.name)


def init_model():
    """Load model vào globals"""
    try:
        G.model = load_from_s3("model.pkl")
        G.scaler = load_from_s3("scaler.pkl")
        G.encoder = load_from_s3("label_encoder.pkl")
        print("[MODEL] Loaded successfully")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        raise
