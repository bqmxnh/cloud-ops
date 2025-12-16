#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
import random
import time
from pathlib import Path

from river import tree, metrics, drift
import mlflow

# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Incremental ARF Retrain")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test-old", required=True)
    ap.add_argument("--test-new", required=True)
    ap.add_argument("--add-ratio", type=float, default=0.4)
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
    # LOAD MODEL + PREPROCESSORS (FROM PRODUCTION)
    # --------------------------------------------------------
    artifact_dir = load_from_registry(MODEL_NAME, MODEL_STAGE)

    model_base = joblib.load(Path(artifact_dir) / "model.pkl")
    scaler = joblib.load(Path(artifact_dir) / "scaler.pkl")
    encoder = joblib.load(Path(artifact_dir) / "label_encoder.pkl")
    FEATURE_ORDER = joblib.load(Path(artifact_dir) / "feature_order.pkl")

    print(f"✔ Trees before retrain: {len(model_base.models)}")
    print(f"✔ Classes: {encoder.classes_}")

    # --------------------------------------------------------
    # LOAD DATA (ALREADY SPLIT + SMOTE)
    # --------------------------------------------------------
    df_train = pd.read_csv(args.train)
    df_old = pd.read_csv(args.test_old)
    df_new = pd.read_csv(args.test_new)

    for df in (df_train, df_old, df_new):
        df["Label"] = encoder.transform(df["Label"])

    print(f"\n[SPLIT]")
    print(f"✔ Train: {len(df_train)}")
    print(f"✔ Test Old: {len(df_old)}")
    print(f"✔ Test New: {len(df_new)}")    

    # ============================================================
    # EVAL BEFORE RETRAIN (ADAPTATION BASELINE)
    # ============================================================
    print("\n[EVAL-BEFORE] Adaptation baseline (Production model)")

    metrics_new_before = evaluate_dataset(
        model_base, scaler, df_new, FEATURE_ORDER, "PRE-ADAPT / TEST_NEW"
    )

    F1_new_before = metrics_new_before["f1"]

    # ============================================================
    # --------------------------------------------------------
    # ADD NEW TREES
    # --------------------------------------------------------
    model = model_base
    model._rng = random.Random()

    num_old = len(model.models)
    num_add = int(num_old * args.add_ratio)

    print(f"\n[MODEL] Adding {num_add} new trees ({int(args.add_ratio*100)}%)")

    def create_new_tree(params):
        """Create new Hoeffding Tree with best params"""
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

    # ============================================================
    # EVALUATION HELPERS (ADDED)
    # ============================================================
    def evaluate_dataset(model, scaler, df, feature_order, name):
        acc = metrics.Accuracy()
        prec = metrics.Precision()
        rec = metrics.Recall()
        f1 = metrics.F1()
        kappa = metrics.CohenKappa()

        tp = tn = fp = fn = 0

        for _, row in df.iterrows():
            x = {k: row[k] for k in feature_order}
            y = row["Label"]

            x_scaled = scaler.transform_one(x)
            y_pred = model.predict_one(x_scaled)

            if y_pred is None:
                continue

            acc.update(y, y_pred)
            prec.update(y, y_pred)
            rec.update(y, y_pred)
            f1.update(y, y_pred)
            kappa.update(y, y_pred)

            if y == 0 and y_pred == 0: tp += 1
            elif y == 1 and y_pred == 1: tn += 1
            elif y == 1 and y_pred == 0: fp += 1
            elif y == 0 and y_pred == 1: fn += 1

        print(f"\n[{name}]")
        print(f"F1-score        : {f1.get():.4f}")
        print(f"Accuracy        : {acc.get():.4f}")
        print(f"Precision       : {prec.get():.4f}")
        print(f"Recall          : {rec.get():.4f}")
        print(f"Cohen's Kappa   : {kappa.get():.4f}")
        print("[CONFUSION MATRIX]")
        print(f"TP: {tp} | FP: {fp}")
        print(f"FN: {fn} | TN: {tn}")

        return {
            "f1": f1.get(),
            "accuracy": acc.get(),
            "precision": prec.get(),
            "recall": rec.get(),
            "kappa": kappa.get(),
        }
    # ============================
    # EVALUATION ON TEST SETS
    # ============================
   
    print("\n[EVAL] Retention (OLD distribution)")
    metrics_old = evaluate_dataset(
        model, scaler, df_old, FEATURE_ORDER, "RETENTION / TEST_OLD"
    )

    print("\n[EVAL] Adaptation (NEW distribution)")
    metrics_new = evaluate_dataset(
        model, scaler, df_new, FEATURE_ORDER, "ADAPTATION / TEST_NEW"
    )


    # --------------------------------------------------------
    # SAVE FOR MLFLOW STAGING
    # --------------------------------------------------------
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(FEATURE_ORDER, "feature_order.pkl")

    print("\n[OK] Retrained model artifacts saved")

    # ============================================================
    # MLFLOW LOG + REGISTER
    # ============================================================
    print("\n[MLFLOW] Logging retrained model...")

    with mlflow.start_run(run_name="arf-incremental-retrain") as run:
        # -------------------------------
        # LOG METRICS
        # -------------------------------
        mlflow.log_metric("retention_ratio", RETENTION)
        mlflow.log_metric("forgetting", FORGETTING)
        mlflow.log_metric("f1_new_before", F1_new_before)
        mlflow.log_metric("f1_new_after", F1_new_after)
        mlflow.log_metric("adaptation_gain", ADAPTATION_GAIN)


        # -------------------------------
        # LOG PARAMS
        # -------------------------------
        mlflow.log_param("add_ratio", args.add_ratio)
        mlflow.log_param("n_trees_before", num_old)
        mlflow.log_param("n_trees_after", len(model.models))
        mlflow.log_param("drift_confidence", BEST_PARAMS["drift_confidence"])
        mlflow.log_param("retrain_type", "incremental_arf")
        mlflow.log_param("source_stage", "Production")

        # -------------------------------
        # LOG ARTIFACTS
        # -------------------------------
        mlflow.log_artifact("model.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("label_encoder.pkl")
        mlflow.log_artifact("feature_order.pkl")

        run_id = run.info.run_id
        print(f"[MLFLOW] Run ID: {run_id}")

    # ============================================================
    # DECISION TO PROMOTE TO STAGING?

    F1_new_after = metrics_new["f1"]
    # Baseline performance before drift (Production)
    F1_BASE = 0.9964

    # Retention: performance trên dữ liệu cũ sau retrain
    F1_retention = metrics_old["f1"]


    # Retention ratio (theo literature)
    RETENTION = F1_retention / F1_BASE

    # Forgetting (absolute performance drop)
    FORGETTING = F1_BASE - F1_retention

    # Adaptation gain
    ADAPTATION_GAIN = F1_new_after - F1_new_before


    # ================================
    # THRESHOLDS
    # ================================
    MIN_RETENTION_RATIO = 0.90      # giữ ≥ 90% kiến thức cũ
    MAX_FORGETTING = 0.10           # cho phép quên ≤ 10%
    MIN_ADAPTATION_GAIN = 0.05     # cải thiện ít nhất 5% so với baseline

    PROMOTE = (
        RETENTION >= MIN_RETENTION_RATIO and
        ADAPTATION_GAIN >= MIN_ADAPTATION_GAIN and
        FORGETTING <= MAX_FORGETTING
    )


    print(
        "\n[DECISION-DRIFT-AWARE]\n"
        f"  Retention   F1 = {RETENTION:.4f}\n"
        f"  Adaptation  F1 = {ADAPTATION_GAIN:.4f}\n"
        f"  Forgetting    = {FORGETTING:.4f}\n"
        f"  => PROMOTE = {PROMOTE}"
    )


    if PROMOTE:
        client = mlflow.tracking.MlflowClient()

        model_uri = f"runs:/{run_id}"

        print("[MLFLOW] Registering model to Registry...")
        result = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )

        version = result.version
        print(f"[MLFLOW] Created model version: {version}")

        print("[MLFLOW] Transitioning model to STAGING...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=True
        )

        print(f"[SUCCESS] Model version {version} promoted to STAGING 🚀")
    else:
        print(
            "[SKIP] Model NOT promoted → "
            "insufficient retention/adaptation or excessive forgetting"
        )



if __name__ == "__main__":
    main()
