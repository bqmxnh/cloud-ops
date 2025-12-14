#!/usr/bin/env python3
import boto3
import argparse
import csv
from boto3.dynamodb.conditions import Attr

# ============================================================
# CONFIG
# ============================================================
TABLE_NAME = "ids_log_system"
REGION = "us-east-1"
MAX_SAMPLES = 400

# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Fetch drift samples from DynamoDB")
    ap.add_argument("--drift-ts", type=int, required=True, help="FIRST_DRIFT_TS (ms)")
    ap.add_argument("--output", default="/data/drift_raw.csv")
    return ap.parse_args()

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    drift_ts = args.drift_ts

    print(f"[INFO] Fetching samples from drift_ts >= {drift_ts}")
    print(f"[INFO] Max samples: {MAX_SAMPLES}")

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)

    items = []
    scan_kwargs = {
        "FilterExpression": (
            Attr("timestamp").gte(drift_ts) &
            Attr("true_label").ne("unknown")
        )
    }

    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    if not items:
        print("[WARN] No samples found after drift")
        print("FETCH_SUCCESS=false")
        exit(10)

    # Sort ASC by timestamp
    items.sort(key=lambda x: int(x["timestamp"]))

    selected = items[:MAX_SAMPLES]

    print(f"[INFO] Selected samples: {len(selected)}")

    # --------------------------------------------------------
    # WRITE CSV
    # --------------------------------------------------------
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["features_json", "true_label"])

        for it in selected:
            writer.writerow([
                it.get("features_json", "{}"),
                str(it.get("true_label", "")).upper()
            ])

    print("=" * 80)
    print("FETCH_SUCCESS=true")
    print(f"SAMPLES={len(selected)}")
    print(f"OUTPUT={args.output}")
    print("=" * 80)
    exit(0)

if __name__ == "__main__":
    main()
