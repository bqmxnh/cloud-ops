#!/usr/bin/env python3
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def print_distribution(df, title):
    print("=" * 80)
    print(title)
    print("-" * 80)
    if "Source" in df.columns:
        print(
            df.groupby(["Source", "Label"])
              .size()
              .reset_index(name="Count")
              .to_string(index=False)
        )
    else:
        print(df["Label"].value_counts())
    print("-" * 80)
    print("TOTAL =", len(df))
    print("=" * 80)


# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        "Preprocess: source-aware split, SMOTE on train only"
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
    # SOURCE-AWARE HOLD-OUT SPLIT
    # ============================================================
    print("[INFO] Splitting by Source to preserve distribution...")
    
    df_drift = df[df["Source"] == "DRIFT"]  # ATTACK only
    df_base_attack = df[(df["Source"] == "BASE") & (df["Label"] == "ATTACK")]
    df_base_benign = df[(df["Source"] == "BASE") & (df["Label"] == "BENIGN")]

    def split(df_part):
        if len(df_part) == 0:
            return pd.DataFrame(), pd.DataFrame()
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

    # --------------------------------------------------------
    # IMPORTANT: DROP Source BEFORE SMOTE
    # --------------------------------------------------------
    print("[INFO] Dropping 'Source' column from both train and test...")
    
    X_train = df_train.drop(columns=["Label", "Source"], errors="ignore")
    y_train = df_train["Label"]
    
    # Drop Source from test too to match train features
    X_test = df_test.drop(columns=["Label", "Source"], errors="ignore")
    y_test = df_test["Label"]

    print(f"[DEBUG] Train features: {X_train.shape[1]} columns")
    print(f"[DEBUG] Test features: {X_test.shape[1]} columns")

    print("\n[INFO] Distribution BEFORE SMOTE (train):")
    print(y_train.value_counts())
    print("-" * 60)

    # --------------------------------------------------------
    # SMOTE (TRAIN ONLY)
    # --------------------------------------------------------
    print("[INFO] Applying SMOTE to training set...")
    smote = SMOTE(random_state=args.seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    df_train_smote = pd.concat(
        [
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res, name="Label")
        ],
        axis=1
    )

    # Reconstruct test without Source
    df_test_clean = pd.concat(
        [
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True)
        ],
        axis=1
    )

    print("\n[INFO] Distribution AFTER SMOTE (train):")
    print(df_train_smote["Label"].value_counts())
    print("-" * 60)

    # --------------------------------------------------------
    # VALIDATION CHECK
    # --------------------------------------------------------
    train_cols = set(df_train_smote.columns)
    test_cols = set(df_test_clean.columns)
    
    if train_cols != test_cols:
        print("\n[ERROR] Column mismatch detected!")
        print(f"Train only: {train_cols - test_cols}")
        print(f"Test only: {test_cols - train_cols}")
        raise RuntimeError("Feature mismatch between train and test!")
    
    print("\n[OK] Train and test have matching features")
    print(f"[OK] Feature count: {len(train_cols) - 1} (+ Label)")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    df_train_smote.to_csv(args.train_out, index=False)
    df_test_clean.to_csv(args.test_out, index=False)

    print("\n" + "=" * 80)
    print("PREPROCESS_SUCCESS=true")
    print(f"TRAIN_SAMPLES={len(df_train_smote)}")
    print(f"TEST_SAMPLES={len(df_test_clean)}")
    print(f"TRAIN_OUT={args.train_out}")
    print(f"TEST_OUT={args.test_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()