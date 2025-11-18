import os, joblib, tempfile, threading, boto3

BUCKET = os.getenv("MODEL_BUCKET", "arf-ids-model-bucket")
VERSION = os.getenv("MODEL_VERSION", "v1.0")

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

model_lock = threading.Lock()
