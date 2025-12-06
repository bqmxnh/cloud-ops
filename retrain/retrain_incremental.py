# retrain_incremental.py

import pandas as pd
import boto3
import io
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow import pyfunc
import os
import time
from river import preprocessing, metrics

# ============================================================
# CONFIG
# ============================================================
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.qmuit.id.vn")
MODEL_NAME = os.getenv("MODEL_NAME", "ARF Baseline Model")
S3_BUCKET = os.getenv("S3_BUCKET", "qmuit-ids-training-data-store")

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("ARF Baseline Training")

s3 = boto3.client("s3")

# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================
def load_production_model():
    client = MlflowClient()
    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    print(f"Loading PRODUCTION version={prod.version}")

    pyf_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    artifacts = pyf_model._model_impl.python_model  # instance of wrapper class

    # Wrapper object lưu artifacts thành attributes
    model = artifacts.model
    scaler = artifacts.scaler
    encoder = artifacts.encoder
    feature_order = artifacts.feature_order
    replay = artifacts.replay

    return model, scaler, encoder, feature_order, replay, prod.version

# ============================================================
# LOAD CSV FROM S3
# ============================================================
def load_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

# ============================================================
# MERGE DATA 70/30
# ============================================================
def merge_datasets(df_drift, df_base, ratio=0.3):
    n_drift = len(df_drift)
    n_base = int(n_drift * ratio / (1 - ratio))
    df_base_sample = df_base.sample(n=min(n_base, len(df_base)), random_state=42)

    df = (
        pd.concat([df_drift, df_base_sample])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    return df

# ============================================================
# STREAM
# ============================================================
def stream(df, order):
    X = df[order]
    y = df["Label"].astype(str)
    for xi, yi in zip(X.to_dict(orient="records"), y):
        yield xi, yi

# ============================================================
# WRAPPER FOR NEW MODEL AFTER RETRAIN
# ============================================================
class ARFIncrementalPredictor(pyfunc.PythonModel):
    def load_context(self, ctx):
        import cloudpickle, os
        p = ctx.artifacts

        self.model = cloudpickle.load(open(os.path.join(p["model"], "model.pkl"), "rb"))
        self.scaler = cloudpickle.load(open(os.path.join(p["scaler"], "scaler.pkl"), "rb"))
        self.encoder = cloudpickle.load(open(os.path.join(p["encoder"], "encoder.pkl"), "rb"))
        self.feature_order = cloudpickle.load(open(os.path.join(p["feature_order"], "feature_order.pkl"), "rb"))
        self.replay = cloudpickle.load(open(os.path.join(p["replay"], "replay.pkl"), "rb"))

    def predict(self, context, df):
        preds = []
        for r in df[self.feature_order].to_dict(orient="records"):
            x = self.scaler.transform_one(r)
            preds.append(self.model.predict_one(x))
        return preds

# ============================================================
# MAIN
# ============================================================
def main():

    model, scaler, encoder, feature_order, replay, prod_ver = load_production_model()

    df_drift = load_s3_csv("drift/drift.csv")
    df_base = load_s3_csv("base/base.csv")
    df = merge_datasets(df_drift, df_base)

    acc = metrics.Accuracy()
    t0 = time.time()

    for xi, yi_lbl in stream(df, feature_order):
        yi = int(encoder.transform([yi_lbl])[0])

        pred = model.predict_one(scaler.transform_one(xi))
        if pred is not None:
            acc.update(yi, pred)

        scaler.learn_one(xi)
        model.learn_one(scaler.transform_one(xi), yi)

    # SAVE local artifacts
    cloudpickle.dump(model, open("model.pkl", "wb"))
    cloudpickle.dump(scaler, open("scaler.pkl", "wb"))
    cloudpickle.dump(encoder, open("encoder.pkl", "wb"))
    cloudpickle.dump(feature_order, open("feature_order.pkl", "wb"))
    cloudpickle.dump(replay, open("replay.pkl", "wb"))

    # LOG MODEL TO MLFLOW
    with mlflow.start_run(run_name="incremental_retrain"):
        mlflow.log_metric("acc_incremental", acc.get())
        mlflow.log_metric("duration", time.time() - t0)

        mlflow.pyfunc.log_model(
            artifact_path="arf_model",
            python_model=ARFIncrementalPredictor(),
            artifacts={
                "model": ".",
                "scaler": ".",
                "encoder": ".",
                "feature_order": ".",
                "replay": ".",
            },
            registered_model_name=MODEL_NAME,
        )

        client = MlflowClient()
        new = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        print(f"New model version {new.version} → STAGING")

    # cleanup
    for f in ["model.pkl", "scaler.pkl", "encoder.pkl", "feature_order.pkl", "replay.pkl"]:
        os.remove(f)


if __name__ == "__main__":
    main()
