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
# EVALUATION HELPER
# ============================================================
def evaluate_dataset(model, scaler, df, feature_order, encoder, name):
    """
    Evaluate model on a dataset with proper confusion matrix calculation.
    ATTACK (class 0) is considered the POSITIVE class for intrusion detection.
    """
    acc = metrics.Accuracy()
    prec = metrics.Precision()
    rec = metrics.Recall()
    f1 = metrics.F1()
    kappa = metrics.CohenKappa()
    
    # Get ATTACK class index (should be 0)
    ATTACK_CLASS = list(encoder.classes_).index('ATTACK')
    
    # Confusion matrix counters
    tp = tn = fp = fn = 0

    for _, row in df.iterrows():
        x = {k: row[k] for k in feature_order}
        y = row["Label"]

        x_scaled = scaler.transform_one(x)
        y_pred = model.predict_one(x_scaled)

        if y_pred is None:
            continue

        # Update River metrics (automatically handle multi-class correctly)
        acc.update(y, y_pred)
        prec.update(y, y_pred)
        rec.update(y, y_pred)
        f1.update(y, y_pred)
        kappa.update(y, y_pred)

        # Confusion matrix with ATTACK as positive class
        # TP: Correctly detected attack
        # TN: Correctly identified benign
        # FP: False alarm (benign predicted as attack)
        # FN: Missed attack (attack predicted as benign)
        
        if y == ATTACK_CLASS and y_pred == ATTACK_CLASS:
            tp += 1  # True Positive - Correctly detected attack
        elif y != ATTACK_CLASS and y_pred != ATTACK_CLASS:
            tn += 1  # True Negative - Correctly identified benign
        elif y != ATTACK_CLASS and y_pred == ATTACK_CLASS:
            fp += 1  # False Positive - False alarm
        elif y == ATTACK_CLASS and y_pred != ATTACK_CLASS:
            fn += 1  # False Negative - Missed attack

    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"{'='*60}")
    print(f"F1-score        : {f1.get():.4f}")
    print(f"Accuracy        : {acc.get():.4f}")
    print(f"Precision       : {prec.get():.4f}")
    print(f"Recall          : {rec.get():.4f}")
    print(f"Cohen's Kappa   : {kappa.get():.4f}")
    print(f"\n[CONFUSION MATRIX - ATTACK as Positive]")
    print(f"                 Predicted")
    print(f"              ATTACK  BENIGN")
    print(f"Actual ATTACK   {tp:4d}    {fn:4d}   (TP/FN)")
    print(f"       BENIGN   {fp:4d}    {tn:4d}   (FP/TN)")
    print(f"{'='*60}\n")

    return {
        "f1": f1.get(),
        "accuracy": acc.get(),
        "precision": prec.get(),
        "recall": rec.get(),
        "kappa": kappa.get(),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

# ============================================================
# TREE CREATION
# ============================================================
def create_new_tree(params):
    """Create new Hoeffding Tree with best params"""
    return tree.HoeffdingTreeClassifier(
        grace_period=params["grace_period"],
        split_criterion=params["split_criterion"],
        leaf_prediction=params["leaf_prediction"],
        binary_split=params["binary_split"],
        max_depth=params["max_depth"]
    )

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    random.seed(args.seed)

    # --------------------------------------------------------
    # LOAD MODEL + PREPROCESSORS (FROM PRODUCTION)
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("[STEP 1] LOADING PRODUCTION MODEL")
    print(f"{'='*60}")
    
    artifact_dir = load_from_registry(MODEL_NAME, MODEL_STAGE)

    model_base = joblib.load(Path(artifact_dir) / "model.pkl")
    scaler = joblib.load(Path(artifact_dir) / "scaler.pkl")
    encoder = joblib.load(Path(artifact_dir) / "label_encoder.pkl")
    FEATURE_ORDER = joblib.load(Path(artifact_dir) / "feature_order.pkl")

    print(f"✔ Trees before retrain: {len(model_base.models)}")
    print(f"✔ Classes: {encoder.classes_}")
    print(f"✔ Feature count: {len(FEATURE_ORDER)}")

    # --------------------------------------------------------
    # LOAD DATA (ALREADY SPLIT + SMOTE)
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("[STEP 2] LOADING DATA")
    print(f"{'='*60}")
    
    df_train = pd.read_csv(args.train)
    df_old = pd.read_csv(args.test_old)
    df_new = pd.read_csv(args.test_new)

    # Encode labels
    for df in (df_train, df_old, df_new):
        df["Label"] = encoder.transform(df["Label"])

    print(f"✔ Train set: {len(df_train)} samples")
    print(f"✔ Test Old (BASE): {len(df_old)} samples")
    print(f"✔ Test New (DRIFT): {len(df_new)} samples")
    
    # Check class distribution
    print(f"\nClass distribution in train:")
    print(df_train["Label"].value_counts().to_dict())

    # ============================================================
    # EVAL BEFORE RETRAIN (BASELINE PERFORMANCE ON DRIFT)
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 3] BASELINE EVALUATION (BEFORE RETRAIN)")
    print(f"{'='*60}")

    metrics_new_before = evaluate_dataset(
        model_base, scaler, df_new, FEATURE_ORDER, encoder, 
        "BASELINE / TEST_NEW (DRIFT DATA)"
    )
    F1_new_before = metrics_new_before["f1"]

    # ============================================================
    # ADD NEW TREES TO MODEL
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 4] EXPANDING MODEL ARCHITECTURE")
    print(f"{'='*60}")
    
    model = model_base
    model._rng = random.Random(args.seed)

    num_old = len(model.models)
    num_add = int(num_old * args.add_ratio)

    print(f"✔ Adding {num_add} new trees ({int(args.add_ratio*100)}% of existing)")
    print(f"✔ Total trees after expansion: {num_old + num_add}")

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

    print(f"✔ Architecture expanded successfully!")

    # ============================================================
    # INCREMENTAL TRAINING
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 5] INCREMENTAL TRAINING")
    print(f"{'='*60}")
    
    t0 = time.time()

    for i, row in df_train.iterrows():
        xi = {k: row[k] for k in FEATURE_ORDER}
        yi = row["Label"]

        xi_scaled = scaler.transform_one(xi)
        model.learn_one(xi_scaled, yi)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            progress = (i + 1) / len(df_train) * 100
            print(
                f"Progress: {i+1}/{len(df_train)} "
                f"({progress:.1f}%) - {elapsed:.1f}s elapsed",
                end="\r"
            )

    elapsed = time.time() - t0
    print(f"\n✔ Training completed in {elapsed:.2f}s")
    print(f"✔ Samples per second: {len(df_train)/elapsed:.1f}")

    # ============================================================
    # EVAL AFTER RETRAIN
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 6] POST-RETRAIN EVALUATION")
    print(f"{'='*60}")

    metrics_old = evaluate_dataset(
        model, scaler, df_old, FEATURE_ORDER, encoder, 
        "RETENTION / TEST_OLD (BASE DATA)"
    )

    metrics_new = evaluate_dataset(
        model, scaler, df_new, FEATURE_ORDER, encoder, 
        "ADAPTATION / TEST_NEW (DRIFT DATA)"
    )

    # ============================================================
    # CALCULATE DRIFT-AWARE METRICS
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 7] CALCULATING DRIFT METRICS")
    print(f"{'='*60}")
    
    F1_new_after = metrics_new["f1"]
    F1_retention = metrics_old["f1"]
    F1_BASE = 0.9964  # Production baseline F1 on original BASE distribution

    # Retention: How much old knowledge is preserved
    RETENTION = F1_retention / F1_BASE if F1_BASE > 0 else 0

    # Forgetting: Absolute performance drop on old distribution
    FORGETTING = F1_BASE - F1_retention

    # Adaptation: Performance gain on new distribution
    ADAPTATION_GAIN = F1_new_after - F1_new_before

    print(f"\n📊 DRIFT-AWARE METRICS:")
    print(f"  Retention Ratio    : {RETENTION:.4f} ({RETENTION*100:.2f}%)")
    print(f"  Forgetting         : {FORGETTING:.4f} ({FORGETTING*100:.2f}%)")
    print(f"  Adaptation Gain    : {ADAPTATION_GAIN:.4f} ({ADAPTATION_GAIN*100:.2f}%)")
    print(f"\n📈 DETAILED PERFORMANCE:")
    print(f"  F1 (BASE before)   : {F1_BASE:.4f}")
    print(f"  F1 (OLD after)     : {F1_retention:.4f}")
    print(f"  F1 (NEW before)    : {F1_new_before:.4f}")
    print(f"  F1 (NEW after)     : {F1_new_after:.4f}")

    # ============================================================
    # SAVE ARTIFACTS
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 8] SAVING ARTIFACTS")
    print(f"{'='*60}")
    
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(FEATURE_ORDER, "feature_order.pkl")

    print(f"✔ Model artifacts saved locally")

    # ============================================================
    # MLFLOW LOGGING
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 9] MLFLOW LOGGING")
    print(f"{'='*60}")

    with mlflow.start_run(run_name="arf-incremental-retrain") as run:
        # Log drift-aware metrics
        mlflow.log_metric("retention_ratio", RETENTION)
        mlflow.log_metric("forgetting", FORGETTING)
        mlflow.log_metric("f1_base_before", F1_BASE)
        mlflow.log_metric("f1_old_after", F1_retention)
        mlflow.log_metric("f1_new_before", F1_new_before)
        mlflow.log_metric("f1_new_after", F1_new_after)
        mlflow.log_metric("adaptation_gain", ADAPTATION_GAIN)
        
        # Log detailed metrics
        for prefix, m in [("old", metrics_old), ("new", metrics_new)]:
            mlflow.log_metric(f"{prefix}_accuracy", m["accuracy"])
            mlflow.log_metric(f"{prefix}_precision", m["precision"])
            mlflow.log_metric(f"{prefix}_recall", m["recall"])
            mlflow.log_metric(f"{prefix}_f1", m["f1"])
            mlflow.log_metric(f"{prefix}_kappa", m["kappa"])
        
        # Log parameters
        mlflow.log_param("add_ratio", args.add_ratio)
        mlflow.log_param("n_trees_before", num_old)
        mlflow.log_param("n_trees_after", len(model.models))
        mlflow.log_param("train_samples", len(df_train))
        mlflow.log_param("test_old_samples", len(df_old))
        mlflow.log_param("test_new_samples", len(df_new))
        mlflow.log_param("drift_confidence", BEST_PARAMS["drift_confidence"])
        mlflow.log_param("retrain_type", "incremental_arf")
        mlflow.log_param("source_stage", MODEL_STAGE)
        mlflow.log_param("seed", args.seed)
        
        # Log artifacts
        mlflow.log_artifact("model.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("label_encoder.pkl")
        mlflow.log_artifact("feature_order.pkl")

        run_id = run.info.run_id
        print(f"✔ MLflow Run ID: {run_id}")

    # ============================================================
    # PROMOTION DECISION
    # ============================================================
    print(f"\n{'='*60}")
    print("[STEP 10] PROMOTION DECISION")
    print(f"{'='*60}")
    
    # Thresholds for promotion
    MIN_RETENTION_RATIO = 0.90      # Keep ≥90% of old knowledge
    MAX_FORGETTING = 0.10           # Allow ≤10% forgetting
    MIN_ADAPTATION_GAIN = 0.05      # Improve ≥5% on new data

    PROMOTE = (
        RETENTION >= MIN_RETENTION_RATIO and
        ADAPTATION_GAIN >= MIN_ADAPTATION_GAIN and
        FORGETTING <= MAX_FORGETTING
    )

    print(f"\n🎯 PROMOTION CRITERIA:")
    print(f"  Retention Ratio   : {RETENTION:.4f} {'✔' if RETENTION >= MIN_RETENTION_RATIO else '✘'} (≥{MIN_RETENTION_RATIO})")
    print(f"  Adaptation Gain   : {ADAPTATION_GAIN:.4f} {'✔' if ADAPTATION_GAIN >= MIN_ADAPTATION_GAIN else '✘'} (≥{MIN_ADAPTATION_GAIN})")
    print(f"  Forgetting        : {FORGETTING:.4f} {'✔' if FORGETTING <= MAX_FORGETTING else '✘'} (≤{MAX_FORGETTING})")
    print(f"\n🚀 DECISION: {'PROMOTE TO STAGING' if PROMOTE else 'DO NOT PROMOTE'}")

    if PROMOTE:
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{run_id}/model.pkl"

        print(f"\n[MLFLOW] Registering model to Registry...")
        result = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run_id,
            description=f"Incremental retrain | Retention: {RETENTION:.3f} | Adaptation: {ADAPTATION_GAIN:.3f}"
        )

        version = result.version
        print(f"✔ Created model version: {version}")

        print(f"[MLFLOW] Transitioning to STAGING...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=True
        )

        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: Model v{version} promoted to STAGING!")
        print(f"{'='*60}\n")
    else:
        reasons = []
        if RETENTION < MIN_RETENTION_RATIO:
            reasons.append(f"Low retention ({RETENTION:.3f} < {MIN_RETENTION_RATIO})")
        if ADAPTATION_GAIN < MIN_ADAPTATION_GAIN:
            reasons.append(f"Insufficient adaptation ({ADAPTATION_GAIN:.3f} < {MIN_ADAPTATION_GAIN})")
        if FORGETTING > MAX_FORGETTING:
            reasons.append(f"Excessive forgetting ({FORGETTING:.3f} > {MAX_FORGETTING})")
        
        print(f"\n{'='*60}")
        print(f"⚠️  MODEL NOT PROMOTED")
        print(f"{'='*60}")
        print(f"Reasons:")
        for r in reasons:
            print(f"  • {r}")
        print(f"\nConsider:")
        print(f"  • Collecting more training data")
        print(f"  • Adjusting add_ratio parameter")
        print(f"  • Using rehearsal/replay techniques")
        print(f"  • Implementing elastic weight consolidation")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()