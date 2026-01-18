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

from river import tree, metrics, drift
import mlflow


# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Incremental Retrain ARF (Production → Staging)")
    ap.add_argument("--train", required=True, help="train_smote.csv")
    ap.add_argument("--test", required=True, help="test_holdout.csv")
    ap.add_argument("--add-ratio", type=float, default=0.4)
    ap.add_argument("--max-trees", type=int, default=20, help="Maximum number of trees to keep")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ============================================================
# CONFIG
# ============================================================
RANDOM_SEED = 42

BEST_PARAMS = {
    "n_models": 13,
    "lambda_value": 7,
    "drift_confidence": 0.0002497786,
    "grace_period": 50,
    "split_criterion": "info_gain",
    "leaf_prediction": "nba",
    "binary_split": True,
    "max_depth": 10,
    "disable_weighted_vote": True
}


# ============================================================
# PRUNING HELPERS
# ============================================================
def get_tree_accuracies(model):
    """
    Get accuracy for each tree in the ensemble.
    Returns list of (tree_idx, accuracy) tuples.
    """
    tree_scores = []
    for i, metric in enumerate(model._metrics):
        accuracy = metric.get() if hasattr(metric, 'get') else 0.0
        tree_scores.append((i, accuracy))
    return tree_scores


def prune_weakest_trees(model, target_count):
    """
    Remove the weakest trees to reach target_count.
    Keeps the trees with highest accuracy scores.
    """
    current_count = len(model.models)
    if current_count <= target_count:
        print(f"[PRUNE] Current: {current_count}, Target: {target_count} - No pruning needed")
        return
    
    num_to_remove = current_count - target_count
    print(f"[PRUNE] Current: {current_count}, Target: {target_count}")
    print(f"[PRUNE] Removing {num_to_remove} weakest trees...")
    
    # Get accuracy scores for each tree
    tree_scores = get_tree_accuracies(model)
    tree_scores.sort(key=lambda x: x[1])  # Sort by accuracy (ascending)
    
    # Get indices of weakest trees to remove
    indices_to_remove = [idx for idx, _ in tree_scores[:num_to_remove]]
    indices_to_remove.sort(reverse=True)  # Remove from highest index first
    
    print("[PRUNE] Weakest trees (to be removed):")
    for idx in indices_to_remove:
        acc = tree_scores[indices_to_remove.index(idx)][1]
        print(f"  Tree {idx}: Accuracy = {acc:.4f}")
    
    # Remove weakest trees
    for idx in indices_to_remove:
        model.models.pop(idx)
        model._metrics.pop(idx)
        model._background.pop(idx)
        model._warning_detectors.pop(idx)
        model._drift_detectors.pop(idx)
        del model._warning_tracker[idx]
        del model._drift_tracker[idx]
    
    # Rebuild the tracking dictionaries with new indices
    new_warning_tracker = {}
    new_drift_tracker = {}
    for i in range(len(model.models)):
        new_warning_tracker[i] = 0
        new_drift_tracker[i] = 0
    model._warning_tracker = new_warning_tracker
    model._drift_tracker = new_drift_tracker
    
    print(f"[PRUNE] Done. Trees remaining: {len(model.models)}")


# ============================================================
# MLFLOW
# ============================================================
MLFLOW_TRACKING_URI = "https://mlflow.qmuit.id.vn"
MODEL_NAME = "ARF Baseline Model"
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

    print(f"✔ Trees before retrain: {len(model_base.models)}")
    print(f"✔ Classes: {encoder.classes_}")

    # --------------------------------------------------------
    # DEEP COPY PRODUCTION MODEL FOR BASELINE COMPARISON
    # --------------------------------------------------------
    print("\n[COPY] Creating production baseline snapshot...")
    model_prod_snapshot = copy.deepcopy(model_base)
    print(f"✔ Baseline snapshot has {len(model_prod_snapshot.models)} trees")

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
    # ADD NEW TREES
    # --------------------------------------------------------
    model = model_base
    model._rng = random.Random()

    num_old = len(model.models)
    num_add = int(num_old * args.add_ratio)

    print(f"\n[MODEL] Adding {num_add} new trees ({int(args.add_ratio * 100)}%)")

    def create_new_tree(params):
        return tree.HoeffdingTreeClassifier(
            grace_period=params["grace_period"],
            split_criterion=params["split_criterion"],
            leaf_prediction=params["leaf_prediction"],
            binary_split=params["binary_split"],
            max_depth=params["max_depth"]
        )

    for _ in range(num_add):
        model.models.append(create_new_tree(BEST_PARAMS))
        model._metrics.append(metrics.Accuracy())
        model._background.append(None)
        model._warning_detectors.append(
            drift.ADWIN(delta=BEST_PARAMS["drift_confidence"])
        )
        model._drift_detectors.append(
            drift.ADWIN(delta=BEST_PARAMS["drift_confidence"])
        )

        idx = len(model.models) - 1
        model._warning_tracker[idx] = 0
        model._drift_tracker[idx] = 0

    print(f"[OK] Total trees after expand: {len(model.models)}")

    # --------------------------------------------------------
    # INCREMENTAL TRAINING
    # --------------------------------------------------------
    print("\n[TRAIN] Incremental learning...")
    t0 = time.time()

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
    # PRUNE WEAKEST TREES IF NEEDED
    # --------------------------------------------------------
    print(f"\n[PRUNE] Checking if pruning is needed...")
    print(f"  Current trees: {len(model.models)}")
    print(f"  Max trees: {args.max_trees}")
    prune_weakest_trees(model, args.max_trees)

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
    with mlflow.start_run(run_name="arf-incremental-retrain") as run:
        mlflow.log_metric("f1_new", f1_new.get())
        mlflow.log_metric("kappa_new", kappa_new.get())
        mlflow.log_metric("f1_prod", f1_prod.get())
        mlflow.log_metric("kappa_prod", kappa_prod.get())
        mlflow.log_metric("f1_gain", f1_gain)
        mlflow.log_metric("kappa_gain", kappa_gain)

        mlflow.log_param("add_ratio", args.add_ratio)
        mlflow.log_param("max_trees", args.max_trees)
        mlflow.log_param("n_trees_before", num_old)
        mlflow.log_param("n_trees_added", num_add)
        mlflow.log_param("n_trees_after_pruning", len(model.models))
        mlflow.log_param("source_stage", "Production")

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
            Key="cooldown/last_retrain_ts.txt",
            Body=ts.encode("utf-8")
        )
        print(f"\n[STATE] last_retrain_ts updated = {ts}")

    with open("/tmp/promote", "w") as f:
        f.write("true" if PROMOTE else "false")

    with open("/tmp/run_id", "w") as f:
        f.write(run_id)

    print(f"\n[DECISION] PROMOTE={PROMOTE}")
    print(f"[OUTPUT] RUN_ID={run_id}")


if __name__ == "__main__":
    main()