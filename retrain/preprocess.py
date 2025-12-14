#!/usr/bin/env python3
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE

# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Preprocess (SMOTE only)")
    ap.add_argument("--input", required=True, help="drift_merged.csv")
    ap.add_argument("--output", default="/data/train_smote.csv")
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

    # Ensure numeric label
    y = df["Label"]
    X = df.drop(columns=["Label"])

    print("[INFO] Class distribution BEFORE SMOTE:")
    print(y.value_counts())
    print("-" * 60)

    smote = SMOTE(random_state=args.seed)
    X_res, y_res = smote.fit_resample(X, y)

    out = pd.concat([X_res, y_res], axis=1)
    out.to_csv(args.output, index=False)

    print("=" * 80)
    print("PREPROCESS_SUCCESS=true")
    print(f"INPUT={args.input}")
    print(f"OUTPUT={args.output}")
    print("Class distribution AFTER SMOTE:")
    print(out["Label"].value_counts())
    print("=" * 80)

if __name__ == "__main__":
    main()
