#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
import random
import time
import copy
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import boto3

from river import neighbors, metrics
import mlflow


# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Incremental Retrain KNN (Production → Staging)")
    ap.add_argument("--train", required=True, help="train_smote.csv")
    ap.add_argument("--test", required=True, help="test_holdout.csv")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ============================================================
# CONFIG
# ============================================================
RANDOM_SEED = 42


# ============================================================
# MLFLOW
# ============================================================
MLFLOW_TRACKING_URI = "https://mlflow.qmuit.id.vn"
MODEL_NAME = "KNN Baseline Model"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_from_registry(model_name, stage):
    uri = f"models:/{model_name}/{stage}"
    print(f"[MLFLOW] Loading model from {uri}")
    return mlflow.artifacts.download_artifacts(uri)


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    random.seed(args.seed)

    # --------------------------------------------------------
    # LOAD PRODUCTION MODEL + PREPROCESSORS
    # --------------------------------------------------------
    artifact_dir = load_from_registry(MODEL_NAME, MODEL_STAGE)

    model_base = joblib.load(Path(artifact_dir) / "model.pkl")
    scaler = joblib.load(Path(artifact_dir) / "scaler.pkl")
    encoder = joblib.load(Path(artifact_dir) / "label_encoder.pkl")
    FEATURE_ORDER = joblib.load(Path(artifact_dir) / "feature_order.pkl")

    print(f"✔ KNN model loaded from Production")
    print(f"✔ Classes: {encoder.classes_}")

    # --------------------------------------------------------
    # DEEP COPY PRODUCTION MODEL FOR BASELINE COMPARISON
    # --------------------------------------------------------
    print("\n[COPY] Creating production baseline snapshot...")
    model_prod_snapshot = copy.deepcopy(model_base)
    print(f"✔ Baseline snapshot created")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    df_train = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)

    df_train["Label"] = encoder.transform(df_train["Label"])
    df_test["Label"] = encoder.transform(df_test["Label"])

    print("\n[SPLIT]")
    print(f"✔ Train: {len(df_train)}")
    print(f"✔ Test : {len(df_test)}")

    # --------------------------------------------------------
    # INCREMENTAL TRAINING (KNN learns incrementally)
    # --------------------------------------------------------
    print("\n[TRAIN] Incremental learning...")
    t0 = time.time()

    model = model_base
    
    for i, row in df_train.iterrows():
        xi = {k: row[k] for k in FEATURE_ORDER}
        yi = row["Label"]

        xi_scaled = scaler.transform_one(xi)
        model.learn_one(xi_scaled, yi)

        if (i + 1) % 500 == 0:
            print(
                f"Progress: {i+1}/{len(df_train)} "
                f"({(i+1)/len(df_train)*100:.1f}%)",
                end="\r"
            )

    print(f"\n[DONE] Training completed in {time.time() - t0:.2f}s")

    # --------------------------------------------------------
    # EVALUATION — NEW MODEL
    # --------------------------------------------------------
    print("\n[EVAL] Evaluating NEW model...")

    f1_new = metrics.F1()
    kappa_new = metrics.CohenKappa()

    for _, row in df_test.iterrows():
        x = {k: row[k] for k in FEATURE_ORDER}
        y = row["Label"]

        y_pred = model.predict_one(scaler.transform_one(x))
        if y_pred is not None:
            f1_new.update(y, y_pred)
            kappa_new.update(y, y_pred)

    print(f"NEW  F1     : {f1_new.get():.4f}")
    print(f"NEW  KAPPA  : {kappa_new.get():.4f}")

    # --------------------------------------------------------
    # EVALUATION — PRODUCTION MODEL (BASELINE SNAPSHOT)
    # --------------------------------------------------------
    print("\n[EVAL] Evaluating PRODUCTION model (baseline)...")

    f1_prod = metrics.F1()
    kappa_prod = metrics.CohenKappa()

    for _, row in df_test.iterrows():
        x = {k: row[k] for k in FEATURE_ORDER}
        y = row["Label"]

        # Use the untouched snapshot for fair comparison
        y_pred = model_prod_snapshot.predict_one(scaler.transform_one(x))
        if y_pred is not None:
            f1_prod.update(y, y_pred)
            kappa_prod.update(y, y_pred)

    print(f"PROD F1     : {f1_prod.get():.4f}")
    print(f"PROD KAPPA  : {kappa_prod.get():.4f}")

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------
    f1_gain = f1_new.get() - f1_prod.get()
    kappa_gain = kappa_new.get() - kappa_prod.get()

    print("\n[COMPARISON]")
    print(f"F1 Gain     : {f1_gain:+.4f}")
    print(f"KAPPA Gain  : {kappa_gain:+.4f}")

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(FEATURE_ORDER, "feature_order.pkl")

    # --------------------------------------------------------
    # MLFLOW LOGGING
    # --------------------------------------------------------
    with mlflow.start_run(run_name="knn-incremental-retrain") as run:
        mlflow.log_metric("f1_new", f1_new.get())
        mlflow.log_metric("kappa_new", kappa_new.get())
        mlflow.log_metric("f1_prod", f1_prod.get())
        mlflow.log_metric("kappa_prod", kappa_prod.get())
        mlflow.log_metric("f1_gain", f1_gain)
        mlflow.log_metric("kappa_gain", kappa_gain)

        mlflow.log_param("source_stage", "Production")
        mlflow.log_param("algorithm", "KNN")

        mlflow.log_artifact("model.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("label_encoder.pkl")
        mlflow.log_artifact("feature_order.pkl")

        run_id = run.info.run_id

    # --------------------------------------------------------
    # PROMOTION DECISION
    # --------------------------------------------------------
    PROMOTE = (
        f1_new.get() > f1_prod.get() and
        kappa_new.get() > kappa_prod.get()
    )

    if PROMOTE:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket="qmuit-training-data-store",
            Key="cooldown/last_retrain_ts_knn.txt",
            Body=ts.encode("utf-8")
        )
        print(f"\n[STATE] last_retrain_ts_knn updated = {ts}")

    with open("/tmp/promote", "w") as f:
        f.write("true" if PROMOTE else "false")

    with open("/tmp/run_id", "w") as f:
        f.write(run_id)

    print(f"\n[DECISION] PROMOTE={PROMOTE}")
    print(f"[OUTPUT] RUN_ID={run_id}")


if __name__ == "__main__":
    main()
