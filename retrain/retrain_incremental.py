import pandas as pd
import boto3
import io
import cloudpickle
import mlflow
from mlflow import pyfunc
from mlflow.tracking import MlflowClient
import time
import os

from river import preprocessing, metrics, forest

# ============================================================
# CONFIG
# ============================================================
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.qmuit.id.vn")
MODEL_NAME = os.getenv("MODEL_NAME", "ARF Baseline Model")   # phải trùng tên khi bạn upload baseline
S3_BUCKET = os.getenv("S3_BUCKET", "qmuit-ids-training-data-store")

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("ARF Baseline Training")

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

    pyf_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

    # Load artifacts
    model         = cloudpickle.load(open(pyf_model._context.artifacts["model"], "rb"))
    scaler        = cloudpickle.load(open(pyf_model._context.artifacts["scaler"], "rb"))
    encoder       = cloudpickle.load(open(pyf_model._context.artifacts["encoder"], "rb"))
    feature_order = cloudpickle.load(open(pyf_model._context.artifacts["feature_order"], "rb"))
    replay        = cloudpickle.load(open(pyf_model._context.artifacts["replay"], "rb"))

    return model, scaler, encoder, feature_order, replay, prod.version


# ============================================================
# Load training data
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
# Stream generator using feature_order
# ============================================================
def record_stream(df, feature_order):
    X = df[feature_order]
    y = df["Label"].astype(str)

    for xi, yi in zip(X.to_dict(orient="records"), y):
        yield xi, yi


# ============================================================
# ARF Predictor wrapper
# ============================================================
class ARFPredictor(pyfunc.PythonModel):
    def load_context(self, context):
        self.model         = cloudpickle.load(open(context.artifacts["model"], "rb"))
        self.scaler        = cloudpickle.load(open(context.artifacts["scaler"], "rb"))
        self.encoder       = cloudpickle.load(open(context.artifacts["encoder"], "rb"))
        self.feature_order = cloudpickle.load(open(context.artifacts["feature_order"], "rb"))
        self.replay        = cloudpickle.load(open(context.artifacts["replay"], "rb"))

    def predict(self, context, df):
        preds = []
        for row in df[self.feature_order].to_dict(orient="records"):
            x_scaled = self.scaler.transform_one(row)
            preds.append(self.model.predict_one(x_scaled))
        return preds


# ============================================================
# MAIN Retrain logic
# ============================================================
def main():

    model, scaler, encoder, feature_order, replay, prod_version = load_production_model()

    df_drift, df_base = load_datasets()
    df = merge_datasets(df_drift, df_base, ratio=0.3)

    acc = metrics.Accuracy()

    print("Starting Incremental Training...")
    t0 = time.time()

    for xi, yi_label in record_stream(df, feature_order):

        yi = int(encoder.transform([yi_label])[0])

        pred = model.predict_one(scaler.transform_one(xi))
        if pred is not None:
            acc.update(yi, pred)

        scaler.learn_one(xi)
        model.learn_one(scaler.transform_one(xi), yi)

    print(f"Incremental Accuracy: {acc.get():.4f}")
    print(f"Duration: {time.time() - t0:.2f}s")

    # Save updated artifacts
    cloudpickle.dump(model,         open("model.pkl", "wb"))
    cloudpickle.dump(scaler,        open("scaler.pkl", "wb"))
    cloudpickle.dump(encoder,       open("encoder.pkl", "wb"))
    cloudpickle.dump(feature_order, open("feature_order.pkl", "wb"))
    cloudpickle.dump(replay,        open("replay.pkl", "wb"))

    # Log new model to MLflow
    with mlflow.start_run(run_name="incremental_retrain"):

        mlflow.log_metric("incremental_acc", acc.get())

        mlflow.pyfunc.log_model(
            artifact_path="arf_model",
            python_model=ARFPredictor(),
            artifacts={
                "model": "model.pkl",
                "scaler": "scaler.pkl",
                "encoder": "encoder.pkl",
                "feature_order": "feature_order.pkl",
                "replay": "replay.pkl"
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

        print(f"New version {new_version} promoted → STAGING")


if __name__ == "__main__":
    main()
