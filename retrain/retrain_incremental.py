# retrain_incremental.py

import pandas as pd
import boto3
import io
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import os
import time
from river import preprocessing, metrics, forest, drift

# ============================================================
# CONFIG
# ============================================================
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.qmuit.id.vn")
MODEL_NAME = os.getenv("MODEL_NAME", "ARF Baseline Model")
S3_BUCKET  = os.getenv("S3_BUCKET", "qmuit-ids-training-data-store")

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("ARF Incremental Retrain")

s3 = boto3.client("s3")

# ============================================================
# BEST PARAMETERS (FIXED)
# ============================================================
BEST_PARAMS = {
    "n_models": 13,
    "lambda_value": 7,
    "drift_confidence": 0.00024977860013662906,
    "grace_period": 50,
    "split_criterion": "info_gain",
    "leaf_prediction": "nba",
    "binary_split": True,
    "max_depth": 10,
    "disable_weighted_vote": True
}

# ============================================================
# LOAD PRODUCTION MODEL (JOBLIB ONLY)
# ============================================================
def load_production_model():
    client = MlflowClient()
    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]

    print(f"Loading PRODUCTION version={prod.version}")

    local = mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/Production")

    model         = joblib.load(os.path.join(local, "model.pkl"))
    scaler        = joblib.load(os.path.join(local, "scaler.pkl"))
    encoder       = joblib.load(os.path.join(local, "label_encoder.pkl"))
    feature_order = joblib.load(os.path.join(local, "feature_order.pkl"))
    replay        = joblib.load(os.path.join(local, "replay.pkl"))

    return model, scaler, encoder, feature_order, replay, prod.version


# ============================================================
# LOAD CSV FROM S3
# ============================================================
def load_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# ============================================================
# MERGE 70/30
# ============================================================
def merge_70_30(df_drift, df_base):
    n_drift = len(df_drift)
    n_base  = int(n_drift * 0.3 / 0.7)
    base_sample = df_base.sample(n=min(n_base, len(df_base)), random_state=42)

    merged = pd.concat([df_drift, base_sample]).sample(frac=1.0, random_state=42)
    print(f"Merged dataset: drift={len(df_drift)}, base={len(base_sample)}, total={len(merged)}")
    return merged.reset_index(drop=True)


# ============================================================
# STREAM
# ============================================================
def stream(df, order):
    X = df[order]
    y = df["Label"].astype(str)
    for xi, yi in zip(X.to_dict(orient="records"), y):
        yield xi, yi


# ============================================================
# MAIN RETRAIN
# ============================================================
def main():

    # --------------------------
    # 1. Load production model
    # --------------------------
    model, scaler, encoder, feature_order, replay, prod_ver = load_production_model()

    # --------------------------
    # 2. Load drift + base datasets
    # --------------------------
    df_drift = load_s3_csv("drift/drift.csv")
    df_base  = load_s3_csv("base/base.csv")
    df = merge_70_30(df_drift, df_base)

    # --------------------------
    # 3. Metrics before/after training
    # --------------------------
    acc  = metrics.Accuracy()
    f1   = metrics.F1()
    prec = metrics.Precision()
    rec  = metrics.Recall()
    kappa = metrics.CohenKappa()

    loss = metrics.LogLoss()

    t0 = time.time()
    print("\nStarting incremental learning...\n")

    # --------------------------
    # 4. Incremental update loop
    # --------------------------
    for xi, yi_lbl in stream(df, feature_order):
        yi = int(encoder.transform([yi_lbl])[0])

        # predict
        pred = model.predict_one(scaler.transform_one(xi))
        proba = model.predict_proba_one(scaler.transform_one(xi))

        if pred is not None:
            acc.update(yi, pred)
            f1.update(yi, pred)
            prec.update(yi, pred)
            rec.update(yi, pred)
            kappa.update(yi, pred)
            loss.update(yi, proba)

        # update
        scaler.learn_one(xi)
        model.learn_one(scaler.transform_one(xi), yi)

    duration = time.time() - t0
    print(f"Incremental retrain completed in {duration:.2f} seconds")
    print(f"Acc={acc.get():.5f}, F1={f1.get():.5f}")

    # --------------------------
    # 5. Save artifacts (joblib)
    # --------------------------
    joblib.dump(model,         "model.pkl")
    joblib.dump(scaler,        "scaler.pkl")
    joblib.dump(encoder,       "encoder.pkl")
    joblib.dump(feature_order, "feature_order.pkl")
    joblib.dump(replay,        "replay.pkl")

    # --------------------------
    # 6. Log into MLflow + register model
    # --------------------------
    with mlflow.start_run(run_name="Incremental Retrain") as run:

        mlflow.log_params(BEST_PARAMS)

        mlflow.log_metrics({
            "acc_incremental": acc.get(),
            "precision": prec.get(),
            "recall": rec.get(),
            "f1": f1.get(),
            "kappa": kappa.get(),
            "logloss": loss.get(),
            "duration": duration,
            "samples_used": len(df),
            "base_ratio": 0.3,
            "drift_ratio": 0.7,
            "prev_version": prod_ver,
        })

        # upload files
        for f in ["model.pkl", "scaler.pkl", "encoder.pkl", "feature_order.pkl", "replay.pkl"]:
            mlflow.log_artifact(f)

        run_id = run.info.run_id

    client = MlflowClient()
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}",
        name=MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered.version,
        stage="Staging",
        archive_existing_versions=False
    )

    print(f"\nNew model version {registered.version} → STAGING\n")

    # cleanup
    for f in ["model.pkl", "scaler.pkl", "encoder.pkl", "feature_order.pkl", "replay.pkl"]:
        os.remove(f)


if __name__ == "__main__":
    main()
