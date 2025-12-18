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

COOLDOWN_HOURS = 24
S3_COOLDOWN_URI = "s3://qmuit-training-data-store/cooldown/last_retrain_ts.txt"
LOCAL_COOLDOWN_FILE = "/tmp/last_retrain_ts.txt"


# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("Drift Detection with ADWIN + Trend Confirmation")
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
    last_retrain_ts = load_last_retrain_ts()
    if last_retrain_ts is None:
        return True

    hours = (datetime.now(timezone.utc) - last_retrain_ts).total_seconds() / 3600
    return hours >= COOLDOWN_HOURS



# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    now = datetime.now(timezone.utc)
    start_ts = int((now - timedelta(hours=args.hours)).timestamp() * 1000)

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

    negative_drift_ts = None

    if args.debug:
        print("=" * 110)
        print("idx | timestamp        | pred -> true | err | width | est   | ADWIN | NEG")
        print("=" * 110)

    for i, it in enumerate(items):
        ts = int(it["timestamp"])
        pred = it["label"]
        true = it["true_label"]

        err = 1 if pred != true else 0
        err_buffer.append(err)
        adwin.update(err)

        neg_drift = False

        if adwin.drift_detected and len(err_buffer) == 2 * K:
            past_err = sum(list(err_buffer)[:K]) / K
            recent_err = sum(list(err_buffer)[K:]) / K
            global_err = adwin.estimation

            if recent_err > past_err and recent_err > global_err:
                neg_drift = True
                negative_drift_ts = ts

        if args.debug:
            print(
                f"{i:03d} | {ts} | "
                f"{pred:6s}->{true:6s} | "
                f"{err}   | "
                f"{int(adwin.width):5d} | "
                f"{adwin.estimation:5.3f} | "
                f"{str(adwin.drift_detected):5s} | "
                f"{neg_drift}"
            )

        if neg_drift:
            break

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------
    print("=" * 80)
    print(f"TOTAL_SAMPLES={len(items)}")

    if negative_drift_ts is not None:
        print("DRIFT=true")
        print("DRIFT_TYPE=NEGATIVE")
        print(f"FIRST_DRIFT_TS={negative_drift_ts}")

        last_retrain_ts = load_last_retrain_ts()

        if last_retrain_ts is not None:
            last_retrain_ms = int(last_retrain_ts.timestamp() * 1000)

            if negative_drift_ts <= last_retrain_ms:
                print("RETRAIN=false")
                print("REASON=DRIFT_ALREADY_HANDLED")
                sys.exit(EXIT_NO_DRIFT)


        if can_retrain():
            print("RETRAIN=true")
            sys.exit(EXIT_DRIFT)
        else:
            print("RETRAIN=false")
            print("REASON=COOLDOWN_ACTIVE")
            sys.exit(EXIT_NO_DRIFT)


    else:
        print("DRIFT=false")
        print("RETRAIN=false")
        sys.exit(EXIT_NO_DRIFT)



if __name__ == "__main__":
    main()
