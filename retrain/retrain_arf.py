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
    ap.add_argument("--add-ratio", type=float, default=None, help="Tree ratio (auto-tuned if not specified)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune", action="store_true", help="Grid search for optimal add-ratio")
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
# GRID SEARCH FOR OPTIMAL ADD-RATIO
# ============================================================
def grid_search_add_ratio(model_base, scaler, encoder, FEATURE_ORDER, df_train, df_test, ratios=None):
    """Test multiple add-ratios and return the best one with composite score.
    
    Composite score prioritizes:
    - F1 score (60% weight)
    - Kappa score (30% weight)  
    - Training time (10% weight, lower is better)
    """
    if ratios is None:
        ratios = [round(x * 0.1, 1) for x in range(1, 11)]  # 0.1, 0.2, ..., 1.0
    
    print("\n[GRID SEARCH] Tuning add-ratio...")
    print(f"Testing ratios: {ratios}\n")
    
    results = []
    
    for ratio in ratios:
        print(f"{'='*70}")
        print(f"Testing add-ratio: {ratio} ({int(ratio*100)}%)")
        print(f"{'='*70}")
        
        # Deep copy for this trial
        model = copy.deepcopy(model_base)
        model._rng = random.Random()
        
        num_old = len(model.models)
        num_add = int(num_old * ratio)
        
        print(f"  Adding {num_add} new trees to {num_old} existing...")
        
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
        
        # Train
        print(f"  Training on {len(df_train)} samples...")
        t0 = time.time()
        for i, row in df_train.iterrows():
            xi = {k: row[k] for k in FEATURE_ORDER}
            yi = row["Label"]
            xi_scaled = scaler.transform_one(xi)
            model.learn_one(xi_scaled, yi)
        train_time = time.time() - t0
        
        # Evaluate
        print(f"  Evaluating on {len(df_test)} samples...")
        f1_score = metrics.F1()
        kappa_score = metrics.CohenKappa()
        
        for _, row in df_test.iterrows():
            x = {k: row[k] for k in FEATURE_ORDER}
            y = row["Label"]
            y_pred = model.predict_one(scaler.transform_one(x))
            if y_pred is not None:
                f1_score.update(y, y_pred)
                kappa_score.update(y, y_pred)
        
        f1_val = f1_score.get()
        kappa_val = kappa_score.get()
        
        print(f"  F1:    {f1_val:.6f}")
        print(f"  KAPPA: {kappa_val:.6f}")
        print(f"  TIME:  {train_time:.2f}s\n")
        
        results.append({
            "ratio": ratio,
            "f1": f1_val,
            "kappa": kappa_val,
            "train_time": train_time,
            "n_trees": len(model.models)
        })
    
    # Normalize metrics for composite scoring
    f1_values = [r["f1"] for r in results]
    kappa_values = [r["kappa"] for r in results]
    time_values = [r["train_time"] for r in results]
    
    f1_min, f1_max = min(f1_values), max(f1_values)
    kappa_min, kappa_max = min(kappa_values), max(kappa_values)
    time_min, time_max = min(time_values), max(time_values)
    
    # Calculate composite score: F1(60%) + Kappa(30%) + InverseTime(10%)
    for r in results:
        f1_norm = (r["f1"] - f1_min) / (f1_max - f1_min) if f1_max > f1_min else 0.5
        kappa_norm = (r["kappa"] - kappa_min) / (kappa_max - kappa_min) if kappa_max > kappa_min else 0.5
        time_norm = (r["train_time"] - time_min) / (time_max - time_min) if time_max > time_min else 0.5
        
        r["score"] = 0.60 * f1_norm + 0.30 * kappa_norm + 0.10 * (1 - time_norm)
    
    # Find best by composite score
    best = max(results, key=lambda x: x["score"])
    
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS (ranked by composite score)")
    print(f"{'='*80}")
    print(f"{'Ratio':<8} {'F1':<10} {'KAPPA':<10} {'TIME':<10} {'SCORE':<10} {'Status':<10}")
    print(f"{'-'*80}")
    
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        marker = " ← BEST" if r == best else ""
        print(f"  {r['ratio']:<7.1f} {r['f1']:<9.6f} {r['kappa']:<9.6f} {r['train_time']:<9.2f}s {r['score']:<9.6f}{marker}")
    print(f"{'='*80}\n")
    
    return best["ratio"], results


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
    # GRID SEARCH OR USE PROVIDED RATIO
    # --------------------------------------------------------
    if args.tune or args.add_ratio is None:
        print("\n[TUNING] Grid searching for optimal add-ratio...")
        best_ratio, tuning_results = grid_search_add_ratio(
            model_base, scaler, encoder, FEATURE_ORDER, df_train, df_test,
            ratios=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        print(f"[TUNING] Optimal add-ratio: {best_ratio}")
        args.add_ratio = best_ratio
    else:
        tuning_results = []
        print(f"[CONFIG] Using provided add-ratio: {args.add_ratio}")

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
    # ADD NEW TREES (FINAL TRAINING WITH BEST RATIO)
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
        mlflow.log_param("n_trees_before", num_old)
        mlflow.log_param("n_trees_after", len(model.models))
        mlflow.log_param("source_stage", "Production")
        
        # Log tuning results if available
        if tuning_results:
            mlflow.log_param("tuning_enabled", True)
            for i, result in enumerate(tuning_results):
                mlflow.log_metric(f"tuning_f1_ratio_{result['ratio']}", result['f1'])
                mlflow.log_metric(f"tuning_kappa_ratio_{result['ratio']}", result['kappa'])

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