import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from river import forest, preprocessing, drift, metrics
from imblearn.over_sampling import SMOTE
import random
import time
import mlflow
from river import tree
from mlflow.tracking import MlflowClient
import os

# ============================================================
# CONFIGURATION
# ============================================================
MLFLOW_TRACKING_URI = "https://mlflow.qmuit.id.vn"
MODEL_NAME = "ARF Baseline Model"
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Argo mounts dataset artifacts into /data/
BASE_CSV = "/data/base.csv"
DRIFT_CSV = "/data/drift.csv"
TEST1 = "/data/test1.csv"
TEST2 = "/data/test2.csv"

# Output directory in container
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Hyperparameters
RANDOM_SEED = 42
ADD_RATIO = 0.4  # 40% new trees 
DRIFT_BASE_RATIO = 7 / 3  # Drift:Base = 7:3
BASE_CLASS_BALANCE = 0.6  # Base 60/40 BENIGN/ATTACK

# Baseline metrics (before retraining)
BASE_F1_RETENTION = 0.9964

# Best hyperparameters
best_params = {
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
# MLFLOW SETUP
# ============================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ============================================================
# LOAD BASE MODEL & PREPROCESSING
# ============================================================
print(f"ADD_RATIO: {ADD_RATIO}")
print("="*70)

def load_from_registry(model_name, stage):
    # Load model artifacts from MLflow Model Registry
    uri = f"models:/{model_name}/{stage}"
    print(f"[MLFLOW] Loading model from registry: {uri}")

    local_dir = mlflow.artifacts.download_artifacts(uri)
    print(f"[MLFLOW] Downloaded to: {local_dir}")
    return local_dir


# -----------------------------------------
# LOAD MODEL FROM REGISTRY INSTEAD OF LOCAL
# -----------------------------------------
artifact_dir = load_from_registry(MODEL_NAME, MODEL_STAGE)

model_base = joblib.load(Path(artifact_dir) / "model.pkl")
scaler = joblib.load(Path(artifact_dir) / "scaler.pkl")
encoder = joblib.load(Path(artifact_dir) / "label_encoder.pkl")
FEATURE_ORDER = joblib.load(Path(artifact_dir) / "feature_order.pkl")

print("[OK] Loaded production model, encoder, scaler from Registry.")

print("[OK] Loaded: model.pkl, scaler.pkl, label_encoder.pkl, feature_order.pkl")
print(f"Encoder classes: {encoder.classes_}")
print(f"Original model trees: {len(model_base.models)}")

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
print(f"\n[DATA] Loading datasets...")

df_base = pd.read_csv(BASE_CSV)
df_drift = pd.read_csv(DRIFT_CSV)
if list(df_drift.columns) != FEATURE_ORDER + ["Label"]:
    df_drift.columns = FEATURE_ORDER + ["Label"]
df_test1 = pd.read_csv(TEST1)
df_test2 = pd.read_csv(TEST2)

# Ensure correct column order
df_base = df_base[FEATURE_ORDER + ["Label"]]
df_drift = df_drift[FEATURE_ORDER + ["Label"]]
df_test1 = df_test1[FEATURE_ORDER + ["Label"]]
df_test2 = df_test2[FEATURE_ORDER + ["Label"]]

# Encode labels
df_base["Label"] = encoder.transform(df_base["Label"])
df_drift["Label"] = encoder.transform(df_drift["Label"])
df_test1["Label"] = encoder.transform(df_test1["Label"])
df_test2["Label"] = encoder.transform(df_test2["Label"])

print(f"[OK] Data loaded:")
print(f"Base data: {len(df_base)} samples")
print(f"Drift data: {len(df_drift)} samples")
print(f"Test1 (retention): {len(df_test1)} samples")
print(f"Test2 (adaptation): {len(df_test2)} samples")

# ============================================================
# Sapmling Strategy
# ============================================================
print("\n" + "="*70)
print("SAMPLING STRATEGY")
print("="*70)

drift_count = len(df_drift)
drift_benign = len(df_drift[df_drift["Label"] == 1])
drift_attack = len(df_drift[df_drift["Label"] == 0])

print(f"\n[DRIFT DATA]")
print(f"  Total      : {drift_count}")
print(f"  BENIGN     : {drift_benign} ({drift_benign/drift_count*100:.1f}%)")
print(f"  ATTACK     : {drift_attack} ({drift_attack/drift_count*100:.1f}%)")

# Calculate base sample size (7:3 ratio)
base_sample_size = int(drift_count / DRIFT_BASE_RATIO)
base_benign_size = int(base_sample_size * BASE_CLASS_BALANCE)
base_attack_size = base_sample_size - base_benign_size

print(f"\n[BASE SAMPLING - {int(BASE_CLASS_BALANCE*100)}/{int((1-BASE_CLASS_BALANCE)*100)} Balance]")
print(f"  Total      : {base_sample_size}")
print(f"  BENIGN     : {base_benign_size}")
print(f"  ATTACK     : {base_attack_size}")

# Sample from base
df_base_benign = df_base[df_base["Label"] == 1].sample(
    n=base_benign_size, replace=False, random_state=RANDOM_SEED
)
df_base_attack = df_base[df_base["Label"] == 0].sample(
    n=base_attack_size, replace=False, random_state=RANDOM_SEED
)
df_base_sampled = pd.concat([df_base_benign, df_base_attack])

print(f"Sampled from base")

# Merge drift + base
df_merged = pd.concat([df_drift, df_base_sampled], axis=0)

merged_benign = len(df_merged[df_merged["Label"] == 1])
merged_attack = len(df_merged[df_merged["Label"] == 0])

print(f"\n[MERGED BEFORE SMOTE]")
print(f"Total: {len(df_merged)}")
print(f"BENIGN: {merged_benign} ({merged_benign/len(df_merged)*100:.1f}%)")
print(f"ATTACK: {merged_attack} ({merged_attack/len(df_merged)*100:.1f}%)")
print(f"Imbalance: {merged_attack/merged_benign:.2f}x more attacks")

# Balance with SMOTE
print(f"\n[CLASS BALANCING WITH SMOTE]")

if merged_benign < merged_attack:
    minority_class, majority_class = 1, 0
    minority_count, majority_count = merged_benign, merged_attack
    target_count = merged_attack
else:
    minority_class, majority_class = 0, 1
    minority_count, majority_count = merged_attack, merged_benign
    target_count = merged_benign

print(f"Minority class: {'BENIGN' if minority_class == 1 else 'ATTACK'} ({minority_count})")
print(f"Majority class: {'BENIGN' if majority_class == 1 else 'ATTACK'} ({majority_count})")

if minority_count >= 2:
    k_neighbors = min(5, minority_count - 1)
    print(f"  Strategy: SMOTE oversample to {target_count} (k_neighbors={k_neighbors})")
    
    sm = SMOTE(
        sampling_strategy={minority_class: target_count},
        random_state=RANDOM_SEED,
        k_neighbors=k_neighbors
    )
    
    X_res, y_res = sm.fit_resample(df_merged[FEATURE_ORDER], df_merged["Label"])
    
    df_final = pd.concat([
        pd.DataFrame(X_res, columns=FEATURE_ORDER),
        pd.Series(y_res, name="Label")
    ], axis=1)
    
    print(f"SMOTE completed")
else:
    print(f"Strategy: Random oversample (not enough samples for SMOTE)")
    df_minority = df_merged[df_merged["Label"] == minority_class]
    df_majority = df_merged[df_merged["Label"] == majority_class]
    
    df_minority_oversampled = df_minority.sample(
        n=majority_count, replace=True, random_state=RANDOM_SEED
    )
    
    df_final = pd.concat([df_majority, df_minority_oversampled])
    print(f"Random oversampling completed")

# Shuffle
df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

final_benign = len(df_final[df_final["Label"] == 1])
final_attack = len(df_final[df_final["Label"] == 0])

print(f"\n[FINAL TRAINING SET]")
print(f"Total: {len(df_final)}")
print(f"BENIGN: {final_benign} ({final_benign/len(df_final)*100:.1f}%)")
print(f"ATTACK: {final_attack} ({final_attack/len(df_final)*100:.1f}%)")
print(f"Balance: {final_attack/final_benign:.2f}x")
print("="*70)

# ============================================================
# CREATE NEW TREE FUNCTION
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
# ADD NEW TREES TO MODEL
# ============================================================
print(f"\n[MODEL] Adding {int(ADD_RATIO*100)}% new trees to ARF...")

model = model_base
model._rng = random.Random()

num_trees_old = len(model.models)
num_add = int(num_trees_old * ADD_RATIO)

print(f"  Old trees: {num_trees_old}")
print(f"  New trees: {num_add}")
print(f"  Total after: {num_trees_old + num_add}")

for i in range(num_add):
    new_tree = create_new_tree(best_params)
    
    model.models.append(new_tree)
    model._metrics.append(metrics.Accuracy())
    model._background.append(None)
    model._warning_detectors.append(
        drift.ADWIN(delta=best_params["drift_confidence"])
    )
    model._drift_detectors.append(
        drift.ADWIN(delta=best_params["drift_confidence"])
    )
    
    tree_idx = len(model.models) - 1
    model._warning_tracker[tree_idx] = 0
    model._drift_tracker[tree_idx] = 0

print(f"[OK] ARF now has {len(model.models)} trees\n")

# ============================================================
# INCREMENTAL TRAINING
# ============================================================
print("="*70)
print("INCREMENTAL TRAINING")
print("="*70)

train_acc = metrics.Accuracy()
train_prec = metrics.Precision()
train_rec = metrics.Recall()
train_f1 = metrics.F1()
train_kappa = metrics.CohenKappa()

print(f"\n[TRAIN] Training with {len(df_final)} samples...")
t0 = time.time()

for idx, row in df_final.iterrows():
    xi = {k: row[k] for k in FEATURE_ORDER}
    yi = row["Label"]
    
    xi_scaled = scaler.transform_one(xi)
    
    pred = model.predict_one(xi_scaled)
    if pred is not None:
        train_acc.update(yi, pred)
        train_prec.update(yi, pred)
        train_rec.update(yi, pred)
        train_f1.update(yi, pred)
        train_kappa.update(yi, pred)
    
    model.learn_one(xi_scaled, yi)
    
    if (idx + 1) % 500 == 0:
        print(f"Progress: {idx+1}/{len(df_final)} samples ({(idx+1)/len(df_final)*100:.1f}%)", end='\r')

train_time = time.time() - t0
print(f"\n\n[DONE] Training completed in {train_time:.2f}s")

print(f"\n[TRAINING METRICS]")
print(f"Accuracy: {train_acc.get():.4f}")
print(f"Precision: {train_prec.get():.4f}")
print(f"Recall: {train_rec.get():.4f}")
print(f"F1-Score: {train_f1.get():.4f} ← PRIMARY METRIC")
print(f"Kappa: {train_kappa.get():.4f}")

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model_comprehensive(model, df_eval, scaler, feature_order, dataset_name):
    print(f"\n{'='*70}")
    print(f"EVALUATING: {dataset_name}")
    print(f"{'='*70}")
    
    acc = metrics.Accuracy()
    prec = metrics.Precision()
    rec = metrics.Recall()
    f1 = metrics.F1()
    kappa = metrics.CohenKappa()
    
    predictions = []
    true_labels = []
    
    for idx, row in df_eval.iterrows():
        x = {k: row[k] for k in feature_order}
        y = row["Label"]
        x_scaled = scaler.transform_one(x)
        pred = model.predict_one(x_scaled)
        
        if pred is not None:
            acc.update(y, pred)
            prec.update(y, pred)
            rec.update(y, pred)
            f1.update(y, pred)
            kappa.update(y, pred)
            
            predictions.append(pred)
            true_labels.append(y)
        
        if (idx + 1) % 500 == 0:
            print(f"Progress: {idx+1}/{len(df_eval)}", end='\r')
    
    print()  # New line after progress
    
    # Calculate confusion matrix components
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    tp = np.sum((predictions == 0) & (true_labels == 0))  # True Positive (Attack detected)
    tn = np.sum((predictions == 1) & (true_labels == 1))  # True Negative (Benign correct)
    fp = np.sum((predictions == 0) & (true_labels == 1))  # False Positive (False alarm)
    fn = np.sum((predictions == 1) & (true_labels == 0))  # False Negative (Missed attack)
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    results = {
        'accuracy': acc.get(),
        'precision': prec.get(),
        'recall': rec.get(),
        'f1': f1.get(),
        'kappa': kappa.get(),
        'far': far, 
        'detection_rate': detection_rate,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    # Print results
    print(f"\n[METRICS]")
    print(f"  F1-Score       : {results['f1']:.4f} ← PRIMARY")
    print(f"  Accuracy       : {results['accuracy']:.4f}")
    print(f"  Precision      : {results['precision']:.4f}")
    print(f"  Recall         : {results['recall']:.4f}")
    print(f"  Cohen's Kappa  : {results['kappa']:.4f}")
    print(f"  False Alarm Rate (FAR) : {results['far']:.4f}")
    print(f"  Detection Rate : {results['detection_rate']:.4f}")
    print(f"\n[CONFUSION MATRIX]")
    print(f"  TP (Attack→Attack)  : {results['tp']:,}")
    print(f"  TN (Benign→Benign)  : {results['tn']:,}")
    print(f"  FP (Benign→Attack)  : {results['fp']:,} ← False Alarms")
    print(f"  FN (Attack→Benign)  : {results['fn']:,} ← Missed Attacks")
    
    return results

# ============================================================
# EVALUATION ON TEST SETS
# ============================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Test 1: Base Retention (Forgetting Test)
results_retention = evaluate_model_comprehensive(
    model, df_test1, scaler, FEATURE_ORDER, "BASE RETENTION TEST (base_eval.csv)"
)

# Test 2: Drift Adaptation
results_adaptation = evaluate_model_comprehensive(
    model, df_test2, scaler, FEATURE_ORDER, "DRIFT ADAPTATION TEST (drift_eval.csv)"
)

# ============================================================
# STABILITY-PLASTICITY ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STABILITY-PLASTICITY ANALYSIS")
print("="*70)

# F1-based scores
f1_retention = results_retention['f1']
f1_adaptation = results_adaptation['f1']

retention_score_f1 = f1_retention / BASE_F1_RETENTION
adaptation_score_f1 = f1_adaptation
forgetting_f1 = BASE_F1_RETENTION - f1_retention

print(f"\n[F1-BASED METRICS - PRIMARY]")
print(f"  F1 Retention       : {f1_retention:.4f}") # mô hình giữ lại kiến thức gốc tốt đến mức nào sau khi huấn luyện tăng dần.
print(f"  F1 Adaptation      : {f1_adaptation:.4f}") # mô hình học kiến thức mới tốt đến mức nào.
print(f"  Retention Score(R) : {retention_score_f1:.4f} (F1_after/F1_before)") # tỉ lệ giữ lại kiến thức gốc.
print(f"  Adaptation Score(A): {adaptation_score_f1:.4f}")  # khả năng học kiến thức mới.
print(f"  Forgetting (F)     : {forgetting_f1:.4f}") # mức độ quên kiến thức gốc.


print(f"\n[IDS OPERATIONAL METRICS]")
print(f" FAR Retention : {results_retention['far']:.4f}")
print(f" FAR Adaptation : {results_adaptation['far']:.4f}")
print(f" Detection Rate(Ret): {results_retention['detection_rate']:.4f}")
print(f" Detection Rate(Ada): {results_adaptation['detection_rate']:.4f}")


# ============================================================
# MLFLOW LOGGING
# ============================================================
print(f"\n[MLFLOW] Logging to MLflow...")

with mlflow.start_run(run_name="incremental_retrain") as run:
    run_id = run.info.run_id
    print(f"  Run ID: {run_id}")
    
    # Save artifacts
    model_path = OUTPUT_DIR / "model.pkl"
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    encoder_path = OUTPUT_DIR / "label_encoder.pkl"
    feat_path = OUTPUT_DIR / "feature_order.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(FEATURE_ORDER, feat_path)
    
    # Log artifacts
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(scaler_path)
    mlflow.log_artifact(encoder_path)
    mlflow.log_artifact(feat_path)
    
    # Log parameters
    mlflow.log_param("add_ratio", ADD_RATIO)
    mlflow.log_param("total_trees", len(model.models))
    mlflow.log_param("old_trees", num_trees_old)
    mlflow.log_param("new_trees", num_add)
    mlflow.log_param("training_samples", len(df_final))
    
    # Log training metrics
    mlflow.log_metric("train_accuracy", train_acc.get())
    mlflow.log_metric("train_precision", train_prec.get())
    mlflow.log_metric("train_recall", train_rec.get())
    mlflow.log_metric("train_f1", train_f1.get())
    mlflow.log_metric("train_kappa", train_kappa.get())
    mlflow.log_metric("train_time_seconds", train_time)
    
    # Log F1-based metrics 
    mlflow.log_metric("f1_retention", f1_retention)
    mlflow.log_metric("f1_adaptation", f1_adaptation)
    mlflow.log_metric("f1_retention_score", retention_score_f1)
    mlflow.log_metric("f1_adaptation_score", adaptation_score_f1)
    mlflow.log_metric("f1_forgetting", forgetting_f1)
    print(f"Logged {len(mlflow.active_run().data.metrics)} metrics")
    print(f"Logged {len(mlflow.active_run().data.params)} parameters")
    
    # Register model
    print(f"\n[MLFLOW] Registering model to Model Registry...")
    
    artifact_uri = mlflow.get_artifact_uri()
    
    result = mlflow.register_model(
        model_uri=artifact_uri,
        name=MODEL_NAME
    )
    
    # Transition to Staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Staging",
        archive_existing_versions=False
    )
    
    print(f" Model registered (version={result.version})")
