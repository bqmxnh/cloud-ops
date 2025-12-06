import pandas as pd
import boto3
import io
from pathlib import Path
from collections import deque
import cloudpickle
import mlflow
from mlflow import pyfunc
from mlflow.tracking import MlflowClient
import time
import os

from river import preprocessing, metrics, forest

# ============================================================
# MLflow config
# ============================================================
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.qmuit.id.vn")
MODEL_NAME = os.getenv("MODEL_NAME", "ARF_IDS_MODEL")
S3_BUCKET = os.getenv("S3_BUCKET", "qmuit-ids-training-data-store")

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("ARF Incremental Retrain (Drift+Base)")

s3 = boto3.client("s3")


# ============================================================
# Load CSV from S3
# ============================================================
def load_s3_csv(key: str):
    print(f"Loading s3://{S3_BUCKET}/{key}")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# ============================================================
# Load Production Model
# ============================================================
def load_production_model():
    client = MlflowClient()
    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]

    print(f"Loading Production Model version={prod.version}")

    model_uri = f"models:/{MODEL_NAME}/Production"
    pyf = mlflow.pyfunc.load_model(model_uri)

    arf_model = cloudpickle.load(open(pyf._context.artifacts["model"], "rb"))
    scaler = cloudpickle.load(open(pyf._context.artifacts["scaler"], "rb"))
    encoder = cloudpickle.load(open(pyf._context.artifacts["label_encoder"], "rb"))

    return arf_model, scaler, encoder, prod.version


# ============================================================
# Load Data FROM S3
# ============================================================
def load_datasets():
    df_drift = load_s3_csv("drift/drift.csv")
    df_base  = load_s3_csv("base/base.csv")

    print(f"Drift dataset: {df_drift.shape}")
    print(f"Base dataset : {df_base.shape}")

    return df_drift, df_base


# ============================================================
# Merge Drift + Base = 70/30
# ============================================================
def merge_datasets(df_drift, df_base, ratio=0.3):
    n_drift = len(df_drift)
    n_base = int(n_drift * ratio / (1 - ratio))

    df_base_sample = df_base.sample(
        n=min(n_base, len(df_base)),
        random_state=42
    )

    merged = (
        pd.concat([df_drift, df_base_sample])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    print(f"Merged dataset = {merged.shape}")
    return merged


# ============================================================
# Convert dataset to river stream
# ============================================================
def record_stream(df):
    X = df.drop(columns=["Label"])
    y = df["Label"].astype(str)

    for xi, yi in zip(X.to_dict(orient="records"), y):
        yield xi, yi


# ============================================================
# ARF Predictor wrapper
# ============================================================
class ARFPredictor(pyfunc.PythonModel):

    def load_context(self, context):
        self.model = cloudpickle.load(open(context.artifacts["model"], "rb"))
        self.scaler = cloudpickle.load(open(context.artifacts["scaler"], "rb"))
        self.encoder = cloudpickle.load(open(context.artifacts["label_encoder"], "rb"))

    def predict(self, context, df):
        preds = []
        for row in df.to_dict(orient="records"):
            x_scaled = self.scaler.transform_one(row)
            preds.append(self.model.predict_one(x_scaled))
        return preds


# ============================================================
# MAIN Incremental Retrain Logic
# ============================================================
def main():

    arf_model, scaler, encoder, prod_version = load_production_model()

    df_drift, df_base = load_datasets()
    df = merge_datasets(df_drift, df_base, ratio=0.3)

    acc = metrics.Accuracy()

    t0 = time.time()
    print("🚀 Starting Incremental Training...")

    for xi, yi_label in record_stream(df):

        yi = int(encoder.transform([yi_label])[0])

        pred = arf_model.predict_one(scaler.transform_one(xi))
        if pred is not None:
            acc.update(yi, pred)

        scaler.learn_one(xi)
        arf_model.learn_one(scaler.transform_one(xi), yi)

    print(f"Incremental Accuracy: {acc.get():.4f}")
    print(f"Done in {time.time() - t0:.2f}s")

    # Save artifacts
    cloudpickle.dump(arf_model, open("model.pkl", "wb"))
    cloudpickle.dump(scaler, open("scaler.pkl", "wb"))
    cloudpickle.dump(encoder, open("label_encoder.pkl", "wb"))

    # Log to MLflow
    with mlflow.start_run(run_name="incremental_retrain"):

        mlflow.log_metric("incremental_acc", acc.get())

        mlflow.pyfunc.log_model(
            artifact_path="arf_model",
            python_model=ARFPredictor(),
            artifacts={
                "model": "model.pkl",
                "scaler": "scaler.pkl",
                "label_encoder": "label_encoder.pkl"
            },
            registered_model_name=MODEL_NAME,
        )

        client = MlflowClient()
        new_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[-1].version

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Staging",
            archive_existing_versions=False
        )

        print(f"New version {new_version} → STAGING")


if __name__ == "__main__":
    main()
