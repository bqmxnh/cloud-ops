import os
import time
import joblib
import tempfile
import boto3
import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient
import threading

from app import globals as G
from app.websocket import broadcast

# ============================================================
# CONFIG
# ============================================================
MLFLOW_TRACKING_URI = "https://mlflow.qmuit.id.vn"
MODEL_NAME = "ARF Baseline Model"
MODEL_STAGE = "Production"

BUCKET = os.getenv("MODEL_BUCKET", "arf-ids-model-bucket")
VERSION = os.getenv("MODEL_VERSION", "v1.0")

CHECK_INTERVAL = 60  # seconds

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


# ============================================================
# Load from MLflow Registry
# ============================================================
def get_production_version():
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        return int(versions[0].version)
    except Exception as e:
        print(f"[MLFLOW] Cannot fetch production version: {e}")
        return None


def load_from_registry(model_name, stage):
    try:
        uri = f"models:/{model_name}/{stage}"
        print(f"[MLFLOW] Loading: {uri}")

        local_dir = mlflow.artifacts.download_artifacts(uri)
        print(f"[MLFLOW] Downloaded to: {local_dir}")

        return Path(local_dir)
    except Exception as e:
        print(f"[MLFLOW] ERROR loading registry: {e}")
        return None


# ============================================================
# Load from S3 (fallback)
# ============================================================
def load_from_s3(name: str):
    try:
        s3 = boto3.client("s3")
        key = f"{VERSION}/{name}"

        print(f"[S3] Downloading s3://{BUCKET}/{key}")

        with tempfile.NamedTemporaryFile() as tmp:
            s3.download_file(BUCKET, key, tmp.name)
            return joblib.load(tmp.name)

    except Exception as e:
        print(f"[S3] FAILED: {e}")
        raise


# ============================================================
# Init model (Registry → S3)
# ============================================================
def init_model():
    print("[INIT] Loading model...")

    new_version = get_production_version()

    with G.model_lock:
        registry_dir = load_from_registry(MODEL_NAME, MODEL_STAGE)

        try:
            if registry_dir:
                G.model         = joblib.load(registry_dir / "model.pkl")
                G.scaler        = joblib.load(registry_dir / "scaler.pkl")
                G.encoder       = joblib.load(registry_dir / "label_encoder.pkl")
                G.FEATURE_ORDER = joblib.load(registry_dir / "feature_order.pkl")

                G.current_model_version = new_version
                G.model_reload_count += 1

                print(f"[MODEL] LOADED from MLflow (v{new_version})")
                return
        except Exception as e:
            print(f"[MODEL] Registry corrupted: {e}")

        # Fallback → S3
        print("[MODEL] Falling back to S3...")
        G.model         = load_from_s3("model.pkl")
        G.scaler        = load_from_s3("scaler.pkl")
        G.encoder       = load_from_s3("label_encoder.pkl")
        G.FEATURE_ORDER = load_from_s3("feature_order.pkl")

        G.model_reload_count += 1
        print("[MODEL] Loaded from S3 successfully!")


# ============================================================
# Background auto-refresh worker
# ============================================================
def auto_refresh_worker():
    """Tự kiểm tra version MLflow mỗi 60s."""
    while True:
        time.sleep(CHECK_INTERVAL)

        try:
            new_version = get_production_version()

            if new_version and new_version != G.current_model_version:
                print(f"\n[MODEL] Detected new Production version {new_version}")
                init_model()

                # Notify UI
                event = {
                    "version": new_version,
                    "reload_count": G.model_reload_count
                }
                try:
                    import asyncio
                    asyncio.create_task(broadcast("model_updated", event))
                except:
                    pass

        except Exception as e:
            print(f"[AUTO-REFRESH ERROR]: {e}")
