#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys

# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Merge drift + base data (7:3, streaming base)")
    ap.add_argument("--drift", required=True, help="drift_raw.csv")
    ap.add_argument("--base", required=True, help="base.csv (large)")
    ap.add_argument("--output", default="/data/drift_merged.csv")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ============================================================
# UTILS
# ============================================================
def read_drift_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames

def explode_features(drift_rows, feature_columns):
    out = []
    for r in drift_rows:
        feats = json.loads(r["features_json"])
        row = {c: feats.get(c, 0) for c in feature_columns}
        row["Label"] = r["true_label"].upper()
        out.append(row)
    return out

def reservoir_sample_csv(path, k, seed=42):
    """
    Reservoir sampling for large CSV.
    Keeps k random rows uniformly.
    """
    random.seed(seed)
    reservoir = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < k:
                reservoir.append(row)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = row

    return reservoir

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    random.seed(args.seed)

    # --------------------------------------------------------
    # LOAD DRIFT (SMALL)
    # --------------------------------------------------------
    print("[INFO] Loading drift data...")
    drift_raw, _ = read_drift_csv(args.drift)

    if not drift_raw:
        print("[ERROR] Drift file is empty")
        sys.exit(1)

    drift_n = len(drift_raw)

    # --------------------------------------------------------
    # COMPUTE 7:3 RATIO
    # --------------------------------------------------------
    total_n = int(drift_n / 0.7)
    base_n = max(0, total_n - drift_n)

    print(f"[INFO] Drift samples : {drift_n}")
    print(f"[INFO] Base samples  : {base_n} (reservoir sampling)")

    # --------------------------------------------------------
    # LOAD BASE (STREAMING)
    # --------------------------------------------------------
    print("[INFO] Streaming base.csv ...")
    base_sample = reservoir_sample_csv(
        args.base,
        k=base_n,
        seed=args.seed
    )

    if not base_sample:
        print("[ERROR] Base sample is empty")
        sys.exit(1)

    base_fields = list(base_sample[0].keys())

    if "Label" not in base_fields:
        print("[ERROR] Base file must contain 'Label' column")
        sys.exit(1)

    feature_cols = [c for c in base_fields if c != "Label"]

    # --------------------------------------------------------
    # EXPLODE DRIFT FEATURES
    # --------------------------------------------------------
    print("[INFO] Exploding drift features_json...")
    drift_rows = explode_features(drift_raw, feature_cols)

    # --------------------------------------------------------
    # MERGE & SHUFFLE
    # --------------------------------------------------------
    merged = drift_rows + base_sample
    random.shuffle(merged)

    # --------------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------------
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(merged)

    print("=" * 80)
    print("MERGE_SUCCESS=true")
    print(f"DRIFT_SAMPLES={drift_n}")
    print(f"BASE_SAMPLES={len(base_sample)}")
    print(f"TOTAL_SAMPLES={len(merged)}")
    print(f"OUTPUT={args.output}")
    print("=" * 80)
    sys.exit(0)

if __name__ == "__main__":
    main()
