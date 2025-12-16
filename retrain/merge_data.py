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
    ap.add_argument("--drift", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--output", default="/data/drift_merged.csv")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ============================================================
# UTILS
# ============================================================
def read_drift_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def explode_features(drift_rows, feature_columns):
    out = []
    for r in drift_rows:
        feats = json.loads(r["features_json"])
        row = {c: feats.get(c, 0) for c in feature_columns}
        row["Label"] = r["true_label"].upper()
        row["Source"] = "DRIFT"
        out.append(row)
    return out

def reservoir_sample_csv(path, k, seed):
    random.seed(seed)
    res = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < k:
                row["Source"] = "BASE"
                res.append(row)
            else:
                j = random.randint(0, i)
                if j < k:
                    row["Source"] = "BASE"
                    res[j] = row
    return res

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    random.seed(args.seed)

    drift_raw = read_drift_csv(args.drift)
    if not drift_raw:
        sys.exit("[ERROR] Empty drift file")

    drift_n = len(drift_raw)
    total_n = int(drift_n / 0.7)
    base_n = total_n - drift_n

    base_sample = reservoir_sample_csv(args.base, base_n, args.seed)
    if not base_sample:
        sys.exit("[ERROR] Empty base sample")

    base_fields = list(base_sample[0].keys())
    if "Source" not in base_fields:
        base_fields.append("Source")

    feature_cols = [c for c in base_fields if c not in ("Label", "Source")]

    drift_rows = explode_features(drift_raw, feature_cols)
    merged = drift_rows + base_sample
    random.shuffle(merged)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(merged)

    print("MERGE_SUCCESS=true")
    print(f"DRIFT={len(drift_rows)} BASE={len(base_sample)} TOTAL={len(merged)}")

if __name__ == "__main__":
    main()
