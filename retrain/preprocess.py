#!/usr/bin/env python3
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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

    # --------------------------------------------------------
    # HOLD-OUT SPLIT (STRATIFIED)
    # --------------------------------------------------------
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["Label"]
    )

    print("[INFO] Distribution BEFORE SMOTE (train):")
    print(df_train["Label"].value_counts())
    print("-" * 60)

    # --------------------------------------------------------
    # SMOTE (TRAIN ONLY)
    # --------------------------------------------------------
    X_train = df_train.drop(columns=["Label", "Source"], errors="ignore")
    y_train = df_train["Label"]

    smote = SMOTE(random_state=args.seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    df_train_smote = pd.concat(
        [
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res, name="Label")
        ],
        axis=1
    )

    print("[INFO] Distribution AFTER SMOTE (train):")
    print(df_train_smote["Label"].value_counts())
    print("-" * 60)

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
