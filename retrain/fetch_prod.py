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
MAX_SAMPLES = 10000


# ============================================================
# ARGS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Fetch drift samples from DynamoDB")
    ap.add_argument("--drift-ts", type=int, required=True, help="FIRST_DRIFT_TS (ms)")
    ap.add_argument("--end-ts", type=int, required=True, help="SECOND_DRIFT_TS (ms)")
    ap.add_argument("--output", default="/data/drift_raw.csv", help="Output CSV file path")
    return ap.parse_args()

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)

    items = []
    scan_kwargs = {
        "FilterExpression": (
            Attr("timestamp").gte(args.drift_ts) &
            Attr("timestamp").lte(args.end_ts) &
            Attr("true_label").ne("unknown")
        )
    }

    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    items.sort(key=lambda x: int(x["timestamp"]))
    items = items[:MAX_SAMPLES]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["features_json", "true_label"])
        for it in items:
            writer.writerow([it.get("features_json", "{}"), it["true_label"]])

    print("FETCH_SUCCESS=true")
    print(f"SAMPLES={len(items)}")


if __name__ == "__main__":
    main()
