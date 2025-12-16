#!/usr/bin/env python3
import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE

# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Preprocess for retention/adaptation")
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", default="/data/train.csv")
    ap.add_argument("--test-old-out", default="/data/test_old.csv")
    ap.add_argument("--test-new-out", default="/data/test_new.csv")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    if "Source" not in df.columns:
        raise RuntimeError("Missing Source column")

    df_base = df[df["Source"] == "BASE"]
    df_drift = df[df["Source"] == "DRIFT"]

    test_old = df_base.sample(frac=0.2, random_state=args.seed)
    test_new = df_drift.sample(frac=0.2, random_state=args.seed)

    train_base = df_base.drop(test_old.index)
    train_drift = df_drift.drop(test_new.index)

    train = pd.concat([train_base, train_drift])

    X = train.drop(columns=["Label", "Source"])
    y = train["Label"]

    smote = SMOTE(random_state=args.seed)
    X_res, y_res = smote.fit_resample(X, y)

    train_smote = pd.concat(
        [pd.DataFrame(X_res, columns=X.columns),
         pd.Series(y_res, name="Label")],
        axis=1
    )

    train_smote.to_csv(args.train_out, index=False)
    test_old.drop(columns=["Source"]).to_csv(args.test_old_out, index=False)
    test_new.drop(columns=["Source"]).to_csv(args.test_new_out, index=False)

    print("PREPROCESS_SUCCESS=true")
    print(f"TRAIN={len(train_smote)} OLD={len(test_old)} NEW={len(test_new)}")

if __name__ == "__main__":
    main()
