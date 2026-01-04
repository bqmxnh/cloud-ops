#!/usr/bin/env python3
import boto3
import argparse
import sys
from datetime import datetime, timedelta, timezone
from collections import deque
from boto3.dynamodb.conditions import Attr
from river.drift import ADWIN
import subprocess


# ============================================================
# CONFIG
# ============================================================
TABLE_NAME = "ids_log_system"
REGION = "us-east-1"

# Exit codes (for Argo DAG control)
EXIT_NO_DRIFT = 10
EXIT_NOT_ENOUGH = 11
EXIT_DRIFT = 0

COOLDOWN_HOURS = 0
S3_COOLDOWN_URI = "s3://qmuit-training-data-store/cooldown/last_retrain_ts.txt"
LOCAL_COOLDOWN_FILE = "/tmp/last_retrain_ts.txt"


# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Drift Detection with ADWIN (2-phase)")
    ap.add_argument("--hours", type=int, default=24, help="Lookback window (hours)")
    ap.add_argument("--delta", type=float, default=0.0002, help="ADWIN delta")
    ap.add_argument("--min-samples", type=int, default=100, help="Minimum labeled samples")
    ap.add_argument("--window", type=int, default=30, help="Trend window size (K)")
    ap.add_argument("--debug", action="store_true", help="Verbose output")
    return ap.parse_args()

def load_last_retrain_ts():
    try:
        subprocess.check_call(
            ["aws", "s3", "cp", S3_COOLDOWN_URI, LOCAL_COOLDOWN_FILE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        with open(LOCAL_COOLDOWN_FILE) as f:
            ts = f.read().strip()
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def can_retrain():
    last = load_last_retrain_ts()
    if last is None:
        return True
    diff = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    return diff >= COOLDOWN_HOURS


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    last_retrain = load_last_retrain_ts()

    now = datetime.now(timezone.utc)
    start_ts = int((now - timedelta(hours=args.hours)).timestamp() * 1000)

    # ======================================
    # CUT LOOKBACK WINDOW BY LAST RETRAIN
    # ======================================
    if last_retrain:
        last_retrain_ms = int(last_retrain.timestamp() * 1000)
        start_ts = max(start_ts, last_retrain_ms)
        print(f"[INFO] Last retrain ts : {last_retrain_ms}")
    print(f"[INFO] Lookback window : {args.hours} hours")
    print(f"[INFO] Start timestamp : {start_ts}")
    print(f"[INFO] ADWIN delta     : {args.delta}")
    print(f"[INFO] Trend window K : {args.window}")
    print("-" * 80)

    # --------------------------------------------------------
    # LOAD DATA FROM DYNAMODB
    # --------------------------------------------------------
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)

    items = []
    scan_kwargs = {
        "FilterExpression": (
            Attr("timestamp").gte(start_ts) &
            Attr("label").ne("unknown") &
            Attr("true_label").ne("unknown")
        )
    }

    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    if len(items) < args.min_samples:
        print(f"[WARN] Not enough labeled samples: {len(items)}")
        print("DRIFT=false")
        sys.exit(EXIT_NOT_ENOUGH)

    # Sort by time (CRITICAL)
    items.sort(key=lambda x: int(x["timestamp"]))

    # --------------------------------------------------------
    # ADWIN + TREND CONFIRMATION
    # --------------------------------------------------------
    adwin = ADWIN(delta=args.delta)
    K = args.window
    err_buffer = deque(maxlen=2 * K)

    in_drift = False
    first_drift_ts = None
    second_drift_ts = None

    for it in items:
        ts = int(it["timestamp"])
        err = 1 if it["label"] != it["true_label"] else 0

        err_buffer.append(err)
        adwin.update(err)

        if adwin.drift_detected and len(err_buffer) == 2 * K:
            past = sum(list(err_buffer)[:K]) / K
            recent = sum(list(err_buffer)[K:]) / K
            est = adwin.estimation

            # Drift start (error ↑)
            if not in_drift and recent > past and recent > est:
                first_drift_ts = ts
                in_drift = True

            # Drift end (error ↓)
            elif in_drift and recent < past and recent < est:
                second_drift_ts = ts
                break

    if first_drift_ts:
        print("DRIFT=true")
        print(f"FIRST_DRIFT_TS={first_drift_ts}")

        if second_drift_ts:
            print(f"SECOND_DRIFT_TS={second_drift_ts}")
        else:
            print(f"SECOND_DRIFT_TS={items[-1]['timestamp']}")

        # ======================================
        # BLOCK RETRAIN IF DRIFT IS OLD
        # ======================================
        if last_retrain:
            last_retrain_ms = int(last_retrain.timestamp() * 1000)

            if first_drift_ts <= last_retrain_ms:
                print("[INFO] Drift already handled by last retrain")
                print(f"[INFO] Drift ts     : {first_drift_ts}")
                print(f"[INFO] Last retrain : {last_retrain_ms}")

                print("RETRAIN=false")
                sys.exit(EXIT_NO_DRIFT)

        # ======================================
        # COOLDOWN CHECK
        # ======================================
        if can_retrain():
            print("RETRAIN=true")
            sys.exit(EXIT_DRIFT)
        else:
            print("RETRAIN=false")
            sys.exit(EXIT_NO_DRIFT)


    print("DRIFT=false")
    print("RETRAIN=false")
    sys.exit(EXIT_NO_DRIFT)


if __name__ == "__main__":
    main()
