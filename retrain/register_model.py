#!/usr/bin/env python3
import argparse
import mlflow

MLFLOW_TRACKING_URI = "https://mlflow.qmuit.id.vn"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model name mapping
MODEL_NAMES = {
    "arf": "ARF Baseline Model",
    "knn": "KNN Baseline Model",
    "hat": "HAT Baseline Model",
}


def parse_args():
    ap = argparse.ArgumentParser("Register & promote model")
    ap.add_argument("--run-id", required=True, help="MLflow Run ID")
    ap.add_argument("--model-type", required=True, choices=["arf", "knn", "hat"],
                    help="Model type: arf, knn, or hat")
    return ap.parse_args()

def main():
    args = parse_args()

    model_name = MODEL_NAMES[args.model_type]
    client = mlflow.tracking.MlflowClient()

    print(f"[MLFLOW] Registering {model_name}")
    print(f"[MLFLOW] Run ID: {args.run_id}")

    result = client.create_model_version(
        name=model_name,
        source=f"runs:/{args.run_id}",
        run_id=args.run_id
    )

    version = result.version
    print(f"[MLFLOW] Created model version {version}")

    print("[MLFLOW] Promoting to STAGING")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"[SUCCESS] {model_name} v{version} promoted to STAGING")

if __name__ == "__main__":
    main()
