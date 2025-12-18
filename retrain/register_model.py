#!/usr/bin/env python3
import argparse
import mlflow

MLFLOW_TRACKING_URI = "https://mlflow.qmuit.id.vn"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "ARF Baseline Model"


def parse_args():
    ap = argparse.ArgumentParser("Register & promote model")
    ap.add_argument("--run-id", required=True)
    return ap.parse_args()

def main():
    args = parse_args()

    client = mlflow.tracking.MlflowClient()

    print(f"[MLFLOW] Registering run {args.run_id}")

    result = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{args.run_id}",
        run_id=args.run_id
    )

    version = result.version
    print(f"[MLFLOW] Created model version {version}")

    print("[MLFLOW] Promoting to STAGING")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"[SUCCESS] Model v{version} promoted to STAGING")

if __name__ == "__main__":
    main()
