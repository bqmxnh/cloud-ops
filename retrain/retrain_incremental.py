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
S3_BUCKET  = os.getenv("S3_BUCKET", "qmuit-training-data-store")

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment("ARF Incremental Retrain")

s3 = boto3.client("s3")

# ============================================================
# FIXED BEST PARAMS (for reference logging)
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
# 1. LOAD PRODUCTION MODEL (JOBLIB)
# ============================================================
def load_production_model():
    client = MlflowClient()
    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]

    print(f"Loading PRODUCTION version={prod.version}")

    local_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}/Production"
    )

    model         = joblib.load(os.path.join(local_dir, "model.pkl"))
    scaler        = joblib.load(os.path.join(local_dir, "scaler.pkl"))
    encoder       = joblib.load(os.path.join(local_dir, "label_encoder.pkl"))
    feature_order = joblib.load(os.path.join(local_dir, "feature_order.pkl"))
    replay        = joblib.load(os.path.join(local_dir, "replay.pkl"))

    return model, scaler, encoder, feature_order, replay, prod.version


# ============================================================
# 2. LOAD CSV FULLY (DRIFT ONLY — small)
# ============================================================
def load_s3_csv(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# ============================================================
# 3. LOAD BASE DATASET FROM S3 USING CHUNKS (NO OOM!)
# ============================================================
def sample_base_from_s3(key, n_samples):
    """
    Đọc file base.csv theo chunks 50k rows và chỉ sample n_samples rows.
    Không bao giờ load toàn bộ file vào RAM.
    """
    print(f"Sampling {n_samples} rows from base dataset (chunked read)...")

    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    stream = io.BytesIO(obj["Body"].read())

    sampled_chunks = []
    chunk_size = 50000

    for chunk in pd.read_csv(stream, chunksize=chunk_size):
        take = min(n_samples, len(chunk))

        sampled = chunk.sample(n=take, random_state=42)
        sampled_chunks.append(sampled)

        n_samples -= take
        if n_samples <= 0:
            break

    df_base = pd.concat(sampled_chunks, ignore_index=True)
    print(f"Sampled base size = {len(df_base)} rows")

    return df_base


# ============================================================
# 4. MERGE DRIFT + BASE (70/30)
# ============================================================
def merge_70_30(df_drift, df_base):
    merged = pd.concat([df_drift, df_base]).sample(frac=1.0, random_state=42)
    print(f"Merged dataset total={len(merged)}")
    return merged.reset_index(drop=True)


# ============================================================
# 5. STREAM GENERATOR
# ============================================================
def stream(df, order):
    X = df[order]
    y = df["Label"].astype(str)
    for xi, yi in zip(X.to_dict(orient="records"), y):
        yield xi, yi


# ============================================================
# 6. MAIN TRAINING LOOP
# ============================================================
def main():

    # Load production model
    model, scaler, encoder, feature_order, replay, prod_ver = load_production_model()

    # Load drift dataset (small)
    df_drift = load_s3_csv("drift/drift.csv")

    # Determine number of base samples needed (70/30)
    n_base_needed = int(len(df_drift) * 0.3 / 0.7)

    # Load base dataset from S3 using chunk sampling (NO OOM)
    df_base = sample_base_from_s3("base/base.csv", n_base_needed)

    # Merge
    df = merge_70_30(df_drift, df_base)

    # Metrics
    acc  = metrics.Accuracy()
    f1   = metrics.F1()
    prec = metrics.Precision()
    rec  = metrics.Recall()
    kappa = metrics.CohenKappa()
    loss = metrics.LogLoss()

    print("\nStarting incremental retrain...\n")
    t0 = time.time()

    # Incremental training loop
    for xi, yi_lbl in stream(df, feature_order):
        yi = int(encoder.transform([yi_lbl])[0])

        x_scaled = scaler.transform_one(xi)
        pred = model.predict_one(x_scaled)
        proba = model.predict_proba_one(x_scaled)

        if pred is not None:
            acc.update(yi, pred)
            f1.update(yi, pred)
            prec.update(yi, pred)
            rec.update(yi, pred)
            kappa.update(yi, pred)
            loss.update(yi, proba)

        scaler.learn_one(xi)
        model.learn_one(x_scaled, yi)

    duration = time.time() - t0
    print(f"Retrain completed in {duration:.2f}s")
    print(f"Final Acc={acc.get():.5f}, F1={f1.get():.5f}")

    # Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(feature_order, "feature_order.pkl")
    joblib.dump(replay, "replay.pkl")

    # MLflow logging
    with mlflow.start_run(run_name="Incremental Retrain") as run:

        mlflow.log_params(BEST_PARAMS)

        mlflow.log_metrics({
            "acc": float(acc.get()),
            "precision": float(prec.get()),
            "recall": float(rec.get()),
            "f1": float(f1.get()),
            "kappa": float(kappa.get()),
            "logloss": float(loss.get()),
            "duration": float(duration),
            "samples_used": float(len(df)),
            "base_ratio": 0.3,
            "drift_ratio": 0.7,
            "prev_version": float(prod_ver),
        })


        for f in ["model.pkl", "scaler.pkl", "label_encoder.pkl",
                  "feature_order.pkl", "replay.pkl"]:
            mlflow.log_artifact(f)

        run_id = run.info.run_id

    # Register new version
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

    print(f"New model version {registered.version} → STAGING")

    # Cleanup
    for f in ["model.pkl", "scaler.pkl", "label_encoder.pkl",
              "feature_order.pkl", "replay.pkl"]:
        os.remove(f)


if __name__ == "__main__":
    main()
