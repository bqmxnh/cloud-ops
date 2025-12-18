#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
import random
import time
from pathlib import Path
from collections import Counter

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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug-n", type=int, default=25, help="Print first N predictions for debugging")
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
# DEBUG UTILS
# ============================================================
def print_df_label_dist(df, title):
    print("=" * 80)
    print(title)
    print("-" * 80)
    cols = [c for c in ["Source", "Label"] if c in df.columns]
    if cols:
        if "Source" in cols:
            print(df.groupby(cols).size().reset_index(name="Count").to_string(index=False))
        else:
            print(df["Label"].value_counts().to_string())
    else:
        print("[WARN] No Label/Source column found")
    print("-" * 80)
    print(f"TOTAL={len(df)}")
    print("=" * 80)

def safe_inverse_label(encoder, y_int):
    # y_int can be int or np.int64
    try:
        return encoder.inverse_transform([int(y_int)])[0]
    except Exception:
        return f"UNK({y_int})"

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

    print("=" * 80)
    print("[MODEL] Loaded artifacts")
    print(f"✔ Trees before retrain: {len(model_base.models)}")
    print(f"✔ Encoder classes_: {list(encoder.classes_)}")
    print("=" * 80)

    # Print mapping (IMPORTANT)
    mapping = {cls: int(encoder.transform([cls])[0]) for cls in encoder.classes_}
    inv_mapping = {v: k for k, v in mapping.items()}
    print("[ENCODER] Mapping:", mapping)
    print("[ENCODER] Inverse :", inv_mapping)
    print("-" * 80)

    # --------------------------------------------------------
    # LOAD DATA (ALREADY SPLIT + SMOTE)
    # --------------------------------------------------------
    df_train = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)

    # Show raw distribution BEFORE encoding
    print_df_label_dist(df_train, "[TRAIN] Raw distribution BEFORE encoding")
    print_df_label_dist(df_test,  "[TEST ] Raw distribution BEFORE encoding")

    # Sanity check: feature columns existence
    missing_feats_train = [c for c in FEATURE_ORDER if c not in df_train.columns]
    missing_feats_test  = [c for c in FEATURE_ORDER if c not in df_test.columns]
    if missing_feats_train:
        print("[ERROR] Missing FEATURE_ORDER columns in TRAIN:", missing_feats_train[:10], f"... total={len(missing_feats_train)}")
        raise RuntimeError("Train is missing required feature columns")
    if missing_feats_test:
        print("[ERROR] Missing FEATURE_ORDER columns in TEST:", missing_feats_test[:10], f"... total={len(missing_feats_test)}")
        raise RuntimeError("Test is missing required feature columns")

    # Encode labels
    df_train["Label"] = encoder.transform(df_train["Label"])
    df_test["Label"] = encoder.transform(df_test["Label"])

    print("=" * 80)
    print("[SPLIT]")
    print(f"✔ Train: {len(df_train)}")
    print(f"✔ Test : {len(df_test)}")
    print("=" * 80)

    # Show encoded distribution
    print("[TRAIN] Encoded label counts:", df_train["Label"].value_counts().to_dict())
    print("[TEST ] Encoded label counts:",  df_test["Label"].value_counts().to_dict())
    print("-" * 80)

    # --------------------------------------------------------
    # ADD NEW TREES
    # --------------------------------------------------------
    model = model_base
    model._rng = random.Random()

    num_old = len(model.models)
    num_add = int(num_old * args.add_ratio)

    print(f"\n[MODEL] Adding {num_add} new trees ({int(args.add_ratio*100)}%)")

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
        model._warning_detectors.append(drift.ADWIN(delta=BEST_PARAMS["drift_confidence"]))
        model._drift_detectors.append(drift.ADWIN(delta=BEST_PARAMS["drift_confidence"]))

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
        yi = int(row["Label"])

        xi_scaled = scaler.transform_one(xi)
        model.learn_one(xi_scaled, yi)

        if (i + 1) % 500 == 0:
            print(f"Progress: {i+1}/{len(df_train)} ({(i+1)/len(df_train)*100:.1f}%)", end="\r")

    print(f"\n[DONE] Training completed in {time.time() - t0:.2f}s")

    # --------------------------------------------------------
    # EVALUATION (HOLD-OUT TEST)
    # --------------------------------------------------------
    print("\n[EVAL] Evaluating on TEST set...")

    acc = metrics.Accuracy()
    prec = metrics.Precision()
    rec = metrics.Recall()
    f1 = metrics.F1()
    kappa = metrics.CohenKappa()

    # Confusion by LABEL NAMES (robust)
    tp = tn = fp = fn = 0

    none_pred = 0
    pred_counter = Counter()
    true_counter = Counter()

    # Print first N samples
    debug_left = args.debug_n

    for idx, row in df_test.iterrows():
        x = {k: row[k] for k in FEATURE_ORDER}
        y = int(row["Label"])

        x_scaled = scaler.transform_one(x)
        y_pred = model.predict_one(x_scaled)

        if y_pred is None:
            none_pred += 1
            continue

        y_pred = int(y_pred)

        acc.update(y, y_pred)
        prec.update(y, y_pred)
        rec.update(y, y_pred)
        f1.update(y, y_pred)
        kappa.update(y, y_pred)

        y_true_name = safe_inverse_label(encoder, y)
        y_hat_name  = safe_inverse_label(encoder, y_pred)

        true_counter[y_true_name] += 1
        pred_counter[y_hat_name]  += 1

        if debug_left > 0:
            print(f"[DBG] idx={idx} y={y}({y_true_name}) pred={y_pred}({y_hat_name})")
            debug_left -= 1

        # Compute confusion matrix using names (no assumption about 0/1)
        if y_true_name == "ATTACK" and y_hat_name == "ATTACK":
            tp += 1
        elif y_true_name == "BENIGN" and y_hat_name == "BENIGN":
            tn += 1
        elif y_true_name == "BENIGN" and y_hat_name == "ATTACK":
            fp += 1
        elif y_true_name == "ATTACK" and y_hat_name == "BENIGN":
            fn += 1

    print("=" * 80)
    print("[EVAL] Sanity counters")
    print(f"None predictions: {none_pred}/{len(df_test)}")
    print("True label counts (name):", dict(true_counter))
    print("Pred label counts (name):", dict(pred_counter))
    print("=" * 80)

    print("\n[RESULTS]")
    print(f"F1-score        : {f1.get():.4f}")
    print(f"Accuracy        : {acc.get():.4f}")
    print(f"Precision       : {prec.get():.4f}")
    print(f"Recall          : {rec.get():.4f}")
    print(f"Cohen's Kappa   : {kappa.get():.4f}")

    print("\n[CONFUSION MATRIX] (ATTACK as positive)")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")

    # --------------------------------------------------------
    # SAVE FOR MLFLOW STAGING
    # --------------------------------------------------------
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(FEATURE_ORDER, "feature_order.pkl")
    print("\n[OK] Retrained model artifacts saved")

    # ============================================================
    # MLFLOW LOG
    # ============================================================
    print("\n[MLFLOW] Logging retrained model...")

    with mlflow.start_run(run_name="arf-incremental-retrain") as run:
        mlflow.log_metric("f1", f1.get())
        mlflow.log_metric("accuracy", acc.get())
        mlflow.log_metric("precision", prec.get())
        mlflow.log_metric("recall", rec.get())
        mlflow.log_metric("kappa", kappa.get())

        mlflow.log_param("add_ratio", args.add_ratio)
        mlflow.log_param("n_trees_before", num_old)
        mlflow.log_param("n_trees_after", len(model.models))
        mlflow.log_param("drift_confidence", BEST_PARAMS["drift_confidence"])
        mlflow.log_param("retrain_type", "incremental_arf")
        mlflow.log_param("source_stage", "Production")

        mlflow.log_artifact("model.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("label_encoder.pkl")
        mlflow.log_artifact("feature_order.pkl")

        run_id = run.info.run_id
        print(f"[MLFLOW] Run ID: {run_id}")

    # ============================================================
    # OUTPUT DECISION FOR ARGO
    # ============================================================
    F1_THRESHOLD = 0.90
    KAPPA_THRESHOLD = 0.90

    PROMOTE = (f1.get() >= F1_THRESHOLD and kappa.get() >= KAPPA_THRESHOLD)

    with open("/tmp/promote", "w") as f:
        f.write("true" if PROMOTE else "false")

    with open("/tmp/run_id", "w") as f:
        f.write(run_id)

    print(f"[DECISION] PROMOTE={PROMOTE}")
    print(f"RUN_ID={run_id}")

if __name__ == "__main__":
    main()
