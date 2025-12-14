#!/usr/bin/env python3
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Preprocess (SMOTE + 80/20 split)")
    ap.add_argument("--input", required=True, help="drift_merged.csv")
    ap.add_argument("--train-out", default="/data/train_smote.csv")
    ap.add_argument("--test-out", default="/data/test_holdout.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
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
    # Split X / y
    # --------------------------------------------------------
    y = df["Label"].astype(str)
    X = df.drop(columns=["Label"])

    print("[INFO] Class distribution BEFORE SMOTE:")
    print(y.value_counts())
    print("-" * 60)

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------
    smote = SMOTE(random_state=args.seed)
    X_res, y_res = smote.fit_resample(X, y)

    df_smote = pd.concat(
        [pd.DataFrame(X_res, columns=X.columns),
         pd.Series(y_res, name="Label")],
        axis=1
    )

    print("[INFO] Class distribution AFTER SMOTE:")
    print(df_smote["Label"].value_counts())
    print("-" * 60)

    # --------------------------------------------------------
    # HOLD-OUT 80/20 SPLIT
    # --------------------------------------------------------
    df_train, df_test = train_test_split(
        df_smote,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df_smote["Label"]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    df_train.to_csv(args.train_out, index=False)
    df_test.to_csv(args.test_out, index=False)

    print("=" * 80)
    print("PREPROCESS_SUCCESS=true")
    print(f"TRAIN_SAMPLES={len(df_train)}")
    print(f"TEST_SAMPLES={len(df_test)}")
    print(f"TRAIN_OUT={args.train_out}")
    print(f"TEST_OUT={args.test_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
