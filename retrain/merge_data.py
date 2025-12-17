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
    ap = argparse.ArgumentParser(
        "Merge drift + base (concept-balanced, drift as anchor)"
    )
    ap.add_argument("--drift", required=True, help="drift_raw.csv")
    ap.add_argument("--base", required=True, help="base.csv (large)")
    ap.add_argument("--output", required=True, help="/data/drift_merged.csv")
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
        row["Source"] = "DRIFT"
        out.append(row)
    return out

def reservoir_sample_csv(path, k, seed=42):
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

def filter_by_label(rows, label, k):
    out = []
    for r in rows:
        if r["Label"].upper() == label:
            r = dict(r)
            r["Source"] = "BASE"
            out.append(r)
            if len(out) >= k:
                break
    return out

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    random.seed(args.seed)

    # --------------------------------------------------------
    # LOAD DRIFT
    # --------------------------------------------------------
    print("[INFO] Loading drift data...")
    drift_raw, _ = read_drift_csv(args.drift)
    if not drift_raw:
        print("[ERROR] Drift file empty")
        sys.exit(1)

    D = len(drift_raw)
    print(f"[INFO] Drift attack samples = {D}")

    # --------------------------------------------------------
    # LOAD BASE (oversample buffer)
    # --------------------------------------------------------
    print("[INFO] Streaming base.csv ...")
    base_buffer = reservoir_sample_csv(
        args.base,
        k=D * 4,   # buffer để đủ benign + attack
        seed=args.seed
    )

    if not base_buffer:
        print("[ERROR] Base sample empty")
        sys.exit(1)

    base_fields = list(base_buffer[0].keys())
    if "Label" not in base_fields:
        print("[ERROR] Base CSV must contain Label column")
        sys.exit(1)

    feature_cols = [c for c in base_fields if c != "Label"]

    # --------------------------------------------------------
    # CONCEPT-BALANCED BASE SAMPLING
    # --------------------------------------------------------
    attack_base = filter_by_label(base_buffer, "ATTACK", D)
    benign_base = filter_by_label(base_buffer, "BENIGN", 2 * D)

    if len(attack_base) < D or len(benign_base) < 2 * D:
        print("[ERROR] Not enough base samples for concept balance")
        sys.exit(1)

    print(f"[INFO] Base ATTACK  = {len(attack_base)}")
    print(f"[INFO] Base BENIGN  = {len(benign_base)}")

    # --------------------------------------------------------
    # EXPLODE DRIFT FEATURES
    # --------------------------------------------------------
    print("[INFO] Exploding drift features...")
    drift_rows = explode_features(drift_raw, feature_cols)

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------
    merged = drift_rows + attack_base + benign_base
    random.shuffle(merged)

    # --------------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------------
    fieldnames = feature_cols + ["Label", "Source"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print("=" * 80)
    print("MERGE_SUCCESS=true")
    print(f"DRIFT_ATTACK={D}")
    print(f"BASE_ATTACK={len(attack_base)}")
    print(f"BASE_BENIGN={len(benign_base)}")
    print(f"TOTAL={len(merged)}")
    print(f"OUTPUT={args.output}")
    print("=" * 80)

if __name__ == "__main__":
    main()
