#!/usr/bin/env python3
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def print_distribution(df, title):
    print("=" * 80)
    print(title)
    print("-" * 80)
    print(
        df.groupby(["Source", "Label"])
          .size()
          .reset_index(name="Count")
          .to_string(index=False)
    )
    print("-" * 80)
    print("TOTAL =", len(df))
    print("=" * 80)


# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        "Preprocess: split first, SMOTE on train only"
    )
    ap.add_argument("--input", required=True, help="/data/drift_merged.csv")
    ap.add_argument("--train-out", required=True, help="/data/train_smote.csv")
    ap.add_argument("--test-out", required=True, help="/data/test_holdout.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    print("[INFO] Loading merged dataset...")
    df = pd.read_csv(args.input)

    if "Label" not in df.columns:
        raise RuntimeError("Missing Label column")


    # ============================================================
    # SOURCE-AWARE HOLD-OUT SPLIT (preserve 1:1:2)
    # ============================================================

    df_drift = df[df["Source"] == "DRIFT"]  # ATTACK only
    df_base_attack = df[(df["Source"] == "BASE") & (df["Label"] == "ATTACK")]
    df_base_benign = df[(df["Source"] == "BASE") & (df["Label"] == "BENIGN")]

    def split(df_part):
        return train_test_split(
            df_part,
            test_size=args.test_size,
            random_state=args.seed
        )

    drift_train, drift_test = split(df_drift)
    base_attack_train, base_attack_test = split(df_base_attack)
    base_benign_train, base_benign_test = split(df_base_benign)

    df_train = pd.concat(
        [drift_train, base_attack_train, base_benign_train],
        ignore_index=True
    )

    df_test = pd.concat(
        [drift_test, base_attack_test, base_benign_test],
        ignore_index=True
    )

    print_distribution(df_train, "[TRAIN] Distribution BEFORE SMOTE (Source x Label)")
    print_distribution(df_test,  "[TEST ] Distribution HOLD-OUT (Source x Label)")

    print("[INFO] Distribution BEFORE SMOTE (train):")
    print(df_train["Label"].value_counts())
    print("-" * 60)

    # ========================================================
    # 🔍 DEBUG: BEFORE SMOTE
    # ========================================================
    print("\n" + "🔍" * 40)
    print("DEBUG: BEFORE SMOTE")
    print("🔍" * 40)
    print(f"df_train columns: {df_train.columns.tolist()}")
    print(f"df_train shape: {df_train.shape}")
    print(f"df_test columns: {df_test.columns.tolist()}")
    print(f"df_test shape: {df_test.shape}")
    print("🔍" * 40 + "\n")

    # --------------------------------------------------------
    # SMOTE (TRAIN ONLY)
    # --------------------------------------------------------
    X_train = df_train.drop(columns=["Label", "Source"], errors="ignore")
    y_train = df_train["Label"]

    # ========================================================
    # 🔍 DEBUG: AFTER DROP
    # ========================================================
    print("\n" + "🔍" * 40)
    print("DEBUG: AFTER DROP (before SMOTE)")
    print("🔍" * 40)
    print(f"X_train columns: {X_train.columns.tolist()}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"'Source' in X_train: {'Source' in X_train.columns}")
    print("🔍" * 40 + "\n")

    smote = SMOTE(random_state=args.seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # ========================================================
    # 🔍 DEBUG: AFTER SMOTE
    # ========================================================
    print("\n" + "🔍" * 40)
    print("DEBUG: AFTER SMOTE")
    print("🔍" * 40)
    print(f"X_res columns: {X_res.shape[1]} features")
    print(f"X_res shape: {X_res.shape}")
    print(f"y_res shape: {y_res.shape}")
    print("🔍" * 40 + "\n")

    df_train_smote = pd.concat(
        [
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res, name="Label")
        ],
        axis=1
    )

    print("=" * 80)
    print("[TRAIN] Distribution AFTER SMOTE (Label only)")
    print(df_train_smote["Label"].value_counts())
    print("=" * 80)

    # ========================================================
    # 🔍 DEBUG: FINAL COMPARISON
    # ========================================================
    print("\n" + "🚨" * 40)
    print("DEBUG: FINAL TRAIN vs TEST COMPARISON")
    print("🚨" * 40)
    print(f"df_train_smote columns: {df_train_smote.columns.tolist()}")
    print(f"df_train_smote shape: {df_train_smote.shape}")
    print(f"'Source' in df_train_smote: {'Source' in df_train_smote.columns}")
    print("-" * 80)
    print(f"df_test columns: {df_test.columns.tolist()}")
    print(f"df_test shape: {df_test.shape}")
    print(f"'Source' in df_test: {'Source' in df_test.columns}")
    print("-" * 80)
    
    train_cols = set(df_train_smote.columns)
    test_cols = set(df_test.columns)
    
    if train_cols != test_cols:
        print("❌ COLUMN MISMATCH DETECTED!")
        print(f"   Columns in TRAIN but NOT in TEST: {train_cols - test_cols}")
        print(f"   Columns in TEST but NOT in TRAIN: {test_cols - train_cols}")
        print("\n💡 THIS IS THE BUG CAUSING YOUR MODEL TO FAIL!")
    else:
        print("✅ Columns match perfectly")
    
    print("🚨" * 40 + "\n")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    df_train_smote.to_csv(args.train_out, index=False)
    df_test.to_csv(args.test_out, index=False)

    print("=" * 80)
    print("PREPROCESS_SUCCESS=true")
    print(f"TRAIN_SAMPLES={len(df_train_smote)}")
    print(f"TEST_SAMPLES={len(df_test)}")
    print(f"TRAIN_OUT={args.train_out}")
    print(f"TEST_OUT={args.test_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()