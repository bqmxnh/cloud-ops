# Cloud-Ops: Adaptive Random Forest IDS with Drift Detection and Continuous Model Retraining

## Project Overview

**Cloud-Ops** is a production-grade machine learning system for **Intrusion Detection System (IDS)** powered by **Adaptive Random Forest (ARF)** algorithms. The system is designed to:

- **Detect network anomalies in real-time** using ARF models with adaptive learning capabilities
- **Monitor concept drift** using statistical drift detection (ADWIN algorithm with trend buffer confirmation)
- **Automatically retrain models** when drift is detected while maintaining a cooldown period
- **Serve predictions via REST API** with WebSocket support for real-time UI updates
- **Manage model lifecycle** through MLflow Registry with seamless version management
- **Maintain reproducibility** through DVC-managed datasets and Kubernetes-based workflow orchestration
- **Provide observability** via Prometheus metrics and structured logging

The system consists of two independent repositories:
- **`cloud-ops`** (this repository) — Contains application code, ML pipelines, Docker builds, and Argo CronWorkflows
- **`manifests-cloud-ops`** — Contains Helm charts, Kubernetes manifests, and Infrastructure as Code (GitOps)

---

## Repository Structure

### Core Directories

```
cloud-ops/
├── api/                          # FastAPI service for real-time inference
│   ├── app/
│   │   ├── main.py              # Application entry point with startup hooks
│   │   ├── inference.py         # Prediction endpoints (/predict)
│   │   ├── model_loader.py      # MLflow registry integration and auto-refresh worker
│   │   ├── metrics.py           # Prometheus metrics exposition
│   │   ├── websocket.py         # Real-time WebSocket broadcasting
│   │   ├── evaluation.py        # Model evaluation endpoints
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   ├── globals.py           # Global state management (model, scaler, encoder)
│   │   └── utils/
│   │       └── preprocess.py    # Feature normalization/scaling utilities
│   ├── Dockerfile               # Container image for API service
│   └── requirements.txt          # Python dependencies (FastAPI, river, mlflow, etc.)
│
├── retrain/                      # Model retraining pipeline components
│   ├── check_drift.py            # Drift detection logic (ADWIN + trend buffer)
│   ├── fetch_prod.py             # Retrieve labeled production data from DynamoDB
│   ├── merge_data.py             # Combine drift data with base dataset (concept-balanced strategy)
│   ├── preprocess.py             # Feature preprocessing with SMOTE oversampling
│   ├── retrain_arf.py            # Incremental model training and evaluation
│   ├── register_model.py          # MLflow registry model registration
│   ├── Dockerfile.retrain_arf    # Container image for Argo job
│   └── requirements.retrain_arf.txt  # ML pipeline dependencies (sklearn, river, mlflow, boto3, dvc)
│
├── workflows/                    # Kubernetes & Argo orchestration
│   └── retrain-workflow.yaml     # CronWorkflow definition (daily 03:00 UTC+7 execution)
│
├── datasets/                     # Data management via DVC
│   └── base/
│       ├── base.csv              # Base training dataset
│       └── base.csv.dvc          # DVC remote pointer (S3)
│
├── config/                       # Configuration templates
│   ├── env.example               # Environment variables reference
│   └── logging.yaml              # Python logging configuration
│
├── mlflow/                       # MLflow tracking & registry service
│   ├── Dockerfile                # MLflow server container
│   └── requirements.txt           # MLflow dependencies
│
├── FLOW_OVERVIEW.txt             # Detailed architecture and data flow documentation
└── README.md                     # This file
```

### Key Dependencies

**API Service (`api/requirements.txt`):**
- `fastapi` — Web framework for REST API
- `uvicorn` — ASGI server
- `river` — Streaming machine learning (Adaptive Random Forest)
- `scikit-learn` — Feature scaling and preprocessing
- `mlflow` — Model registry and experiment tracking
- `boto3` — AWS S3 integration
- `prometheus_client` — Metrics exposition
- `joblib` — Model serialization

**Retrain Pipeline (`retrain/requirements.retrain_arf.txt`):**
- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — Feature preprocessing and evaluation metrics
- `imbalanced-learn` — SMOTE oversampling
- `river` — ARF model training
- `mlflow` — Experiment logging
- `boto3`, `s3fs` — S3 data access
- `dvc`, `dvc-s3` — Dataset versioning and retrieval

---

## Architecture & Data Flow

### 1. Real-Time Inference Pipeline

**Initialization Phase:**
```
app/main.py startup
  ├─ init_model()
  │  ├─ Fetch model Production version from MLflow Registry
  │  ├─ Fallback to S3 (arf-ids-model-bucket) if registry unavailable
  │  └─ Load: model.pkl, scaler.pkl, label_encoder.pkl, feature_order.pkl
  │
  └─ Start auto_refresh_worker thread (checks for model updates every CHECK_INTERVAL)
```

**Prediction Flow:**
```
POST /predict (app/inference.py)
  ├─ Parse request: FlowSchema (features as input)
  ├─ Normalize features using loaded scaler
  ├─ predict_proba_confident()
  │  ├─ Query ARF for probability from low-entropy trees only
  │  └─ Fallback to full ensemble if needed
  ├─ Decode prediction using label_encoder
  ├─ Record metrics (prediction_requests_total, prediction_latency_ms)
  └─ Return: {"flow_id", "prediction", "confidence", "latency_ms"}
```

**WebSocket Real-Time Updates:**
- Successful predictions are broadcasted to all connected WebSocket clients via `app/websocket.py`
- Model update events include version number and reload count
- Clients can subscribe to real-time prediction feed and model change notifications

**Metrics & Observability:**
- Prometheus metrics exposed at `/metrics` endpoint (app/metrics.py)
- Counters: `prediction_requests_total`, `model_reloads_total`
- Histograms: `prediction_latency_ms`
- Gauges: `model_version`, `model_reload_count`, `active_websocket_connections`

---

### 2. Data Collection & Labeling

**Production Data Logging:**
- Every prediction from the API is logged with metadata (flow_id, features, timestamp)
- Post-deployment labels (true_label) are collected from external sources and stored in **DynamoDB table `ids_log_system`**
- This labeled data forms the basis for drift detection and model retraining

**Dataset Management:**
- Base training dataset (`datasets/base/base.csv`) is versioned using **DVC**
- DVC remote is configured to **S3 bucket `qmuit-training-data-store`**
- `retrain/fetch_prod.py` queries DynamoDB to retrieve recent labeled samples for drift analysis

---

### 3. Drift Detection & Retraining Decision

**CronWorkflow Execution (Daily at 03:00 UTC+7):**

1. **Check Drift Task** (`retrain/check_drift.py`):
   - Retrieves samples with timestamp ≥ lookback window (e.g., last 24 hours)
   - Filters out samples with unknown/uncertain labels
   - Applies **ADWIN drift detector** with `delta` parameter (default: 0.002)
   - Uses **trend buffer** (size: 2000 samples) to confirm negative drift (increasing error rate)
   - Outputs:
     - `DRIFT=true/false` — Whether drift is detected
     - `DRIFT_TYPE=negative/...` — Type of drift
     - `FIRST_DRIFT_TS=<timestamp>` — When drift started
   - Exit codes determine whether to proceed with retraining

2. **Cooldown Mechanism:**
   - Prevents excessive retraining by checking `S3://qmuit-training-data-store/cooldown/last_retrain_ts.txt`
   - Ensures minimum interval between retraining cycles
   - Automatically updated when retrain pipeline completes successfully

---

### 4. Model Retraining Pipeline (Argo CronWorkflow)

The complete retraining workflow executes the following sequential tasks:

**Task 1: fetch-drift**
- Triggered only if `check-drift` returns `retrain=true`
- Executes `retrain/fetch_prod.py`:
  - Queries DynamoDB for up to 400 samples from `FIRST_DRIFT_TS` onward
  - Saves to container path `/data/drift_raw.csv`
  - Prepares artifact for downstream tasks

**Task 2: save-drift-dvc**
- Initializes DVC in container (no-scm mode)
- Copies drift data to `datasets/drift/drift_<timestamp>.csv`
- Executes `dvc add` and `dvc push` to archive drift data to S3 remote
- Maintains historical record of all detected drift samples

**Task 3: pull-base**
- Uses DVC to pull `datasets/base/base.csv` from S3 remote
- Ensures consistent base dataset version across training runs

**Task 4: merge-data** (`retrain/merge_data.py`)
- Combines drift and base datasets using **concept-balanced strategy**:
  - Drift samples serve as anchor (100%)
  - Adds proportional benign and attack samples from base dataset
  - Assigns `Source=DRIFT` or `Source=BASE` label for traceability
- Output: Merged dataset for training

**Task 5: preprocess** (`retrain/preprocess.py`)
- Performs stratified train/test split (default: 80/20)
- Applies **SMOTE oversampling** to training set (handles class imbalance)
- Generates outputs:
  - `train_smote.csv` — Balanced training data
  - `test_holdout.csv` — Hold-out test set

**Task 6: retrain** (`retrain/retrain_arf.py`)
- Loads current Production model from MLflow Registry and creates a baseline snapshot for comparison
- Incremental training configuration:
  - `add_ratio=0.4` — Adds 40% new trees to existing ensemble
  - Uses original feature scaler for consistency
  - Trains on preprocessed balanced dataset
- Evaluation metrics computed on hold-out test set:
  - **NEW model**: F1-Score and Cohen's Kappa on trained model
  - **PROD model**: F1-Score and Cohen's Kappa on production baseline snapshot (for comparison)
  - Calculates gains: `F1_gain = F1_new - F1_prod` and `Kappa_gain = Kappa_new - Kappa_prod`
- Results logged to MLflow run (`arf-incremental-retrain`)
- **Promotion Decision**: If F1_new > F1_prod AND Kappa_new > Kappa_prod (relative improvement):
  - Writes `promote=true` to `/tmp/promote`
  - Updates cooldown timestamp in S3
  - If promotion not approved: `promote=false`

**Task 7: register-model** (`retrain/register_model.py`)
- Triggered only if `promote=true`
- Creates new model version in MLflow Registry ("ARF Baseline Model")
- Sets stage to **Staging** (archives previous versions)
- Prepares for manual or automated promotion to Production

---

### 5. Model Update & Deployment

**Model Lifecycle Management:**

```
MLflow Registry (Source of Truth)
├─ ARF Baseline Model
│  ├─ Version N (Stage: Production)
│  ├─ Version N+1 (Stage: Staging) ← New retrain output
│  └─ Versions N-1, N-2 (Stage: Archived)
```

**API Auto-Refresh Mechanism:**
- Background worker thread (`app/model_loader.py::auto_refresh_worker`) runs every `CHECK_INTERVAL` seconds
- Polls MLflow Registry for Production model version
- When new Production version detected:
  - Downloads model artifacts
  - Updates global state (model, scaler, encoder)
  - Increments `MODEL_RELOAD_COUNT`
  - Broadcasts `model_updated` event via WebSocket
  - Updates Prometheus gauge `model_version`

**CI/CD Integration (with manifests-cloud-ops):**

1. **Build Pipeline** (`build-api.yml` in `.github/workflows/`):
   - Triggers on code push or manual dispatch
   - Builds Docker image for API service
   - Tags image: `bqmxnh/arf-ids-api:YYYYMMDDHHMMSS`
   - Pushes to container registry
   - Triggers update workflow in `manifests-cloud-ops` repo

2. **Deployment Pipeline** (`update-helm.yml` in `manifests-cloud-ops`):
   - Receives trigger via `workflow_dispatch` or `repository_dispatch`
   - Updates Helm values file: `values/api/values.yaml` with new image tag
   - Commits and pushes changes to Git

3. **GitOps Reconciliation** (Argo CD):
   - Continuously monitors `manifests-cloud-ops` main branch
   - Detects Helm values changes
   - Automatically reconciles Kubernetes cluster state
   - Deploys new container image without manual intervention
   - Auto-sync + self-healing ensures alignment between Git and cluster

---


## System Data Flow (End-to-End)

```
1. INITIALIZATION
   API Service Startup
   ├─ Load ARF model from MLflow Registry → /predict ready
   ├─ Start auto_refresh_worker
   └─ Expose Prometheus metrics

2. REAL-TIME INFERENCE
   User/Client Request
   ├─ POST /predict with network flow
   ├─ API normalizes features
   ├─ ARF returns prediction + confidence
   ├─ Log to DynamoDB (flow_id, features, prediction, timestamp)
   ├─ Broadcast via WebSocket to UI
   └─ Return JSON response + HTTP 200

3. LABELS & MONITORING
   External System
   ├─ Collect true labels for flows (post-deployment)
   ├─ Write to DynamoDB (update true_label field)
   ├─ Dashboard queries DynamoDB for accuracy metrics

4. DAILY DRIFT CHECK (03:00 UTC+7)
   CronWorkflow Trigger
   ├─ Query DynamoDB for samples (last 24h)
   ├─ Apply ADWIN drift detector + trend buffer
   ├─ If DRIFT=true && cooldown expired → continue
   └─ Otherwise → exit (no retrain)

5. RETRAINING PIPELINE
   If Drift Detected
   ├─ Fetch production samples from DynamoDB (drift_raw.csv)
   ├─ Save to DVC + push to S3
   ├─ Pull base dataset (base.csv)
   ├─ Merge with concept-balanced strategy
   ├─ Apply SMOTE + stratified split
   ├─ Incremental train on ARF (add_ratio=0.4)
   ├─ Evaluate both NEW and PROD models on hold-out set
   ├─ Compare metrics: F1_gain = F1_new - F1_prod, Kappa_gain = Kappa_new - Kappa_prod
   ├─ Log metrics to MLflow
   ├─ If F1_new > F1_prod AND Kappa_new > Kappa_prod → register new version (Staging)
   └─ Update cooldown timestamp

6. MODEL DEPLOYMENT
   MLflow Registry Update
   ├─ Manual or automated promotion to Production
   ├─ API auto_refresh_worker detects version change
   ├─ Downloads new model artifacts
   ├─ Swaps in-memory model
   └─ Broadcasts model_updated event via WebSocket

7. INFRASTRUCTURE DEPLOYMENT (Optional)
   Code Push to cloud-ops
   ├─ GitHub Actions build-api.yml triggers
   ├─ Build + push container image
   ├─ Trigger manifests-cloud-ops update workflow
   ├─ Helm values updated with new image tag
   ├─ Argo CD detects Git change
   ├─ Auto-sync reconciles cluster state
   └─ New API pods rolled out
```

---

## API Endpoints

### Prediction Endpoint
```
POST /predict
Content-Type: application/json

{
  "flow_id": "flow_12345",
  "features": [1.2, 3.4, 5.6, ...]  // 20 numeric features
}

Response (200 OK):
{
  "flow_id": "flow_12345",
  "prediction": "attack",
  "confidence": 0.95,
  "latency_ms": 12.34
}
```

### Metrics Endpoint
```
GET /metrics

Returns Prometheus metrics:
- prediction_requests_total{endpoint="/predict"} 15234
- prediction_latency_ms_bucket{le="50"} 14500
- prediction_latency_ms_bucket{le="100"} 14950
- model_version 3
- model_reload_count 2
```

### Health Check
```
GET /

Response:
{
  "status": "running",
  "version": "9.0"
}
```

### WebSocket (Real-Time Updates)
```
WS /ws

Receives JSON messages:
{"type": "prediction", "flow_id": "flow_12345", "prediction": "benign"}
{"type": "model_updated", "version": 3, "reload_count": 2}
```

---

## Monitoring & Observability

### Prometheus Metrics
- **prediction_requests_total** — Total predictions served
- **prediction_latency_ms** — Prediction response time histogram
- **model_version** — Current model version in production
- **model_reload_total** — Number of times model was reloaded
- **active_websocket_connections** — Connected UI clients

### Logging
- **Log Location:** Configured via `config/logging.yaml`
- **Log Levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Key Events:**
  - Model initialization and auto-refresh
  - Prediction requests (with latency)
  - WebSocket client connections
  - Drift detection results
  - Model registration and promotion

### DynamoDB Monitoring
- Monitor `ids_log_system` table capacity and latency
- Query patterns: Recent samples for drift detection
- Labeling rate: Percentage of predictions that receive true labels

---

## Development & Local Testing

### Prerequisites
```bash
# Python 3.10+ (for local testing)
python --version

# Docker and Docker Compose
docker --version

# DVC (for data versioning)
pip install dvc dvc-s3

# AWS CLI (for S3 access)
aws --version
```

### Running API Locally
```bash
cd api
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_BUCKET=local-model-bucket
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Retrain Pipeline Locally
```bash
cd retrain
pip install -r requirements.retrain_arf.txt
python check_drift.py     # Drift detection
python fetch_prod.py      # Fetch drift data
python merge_data.py      # Merge datasets
python preprocess.py      # Apply SMOTE
python retrain_arf.py     # Train model
python register_model.py  # Register to MLflow
```

---

## Contributing & Best Practices

1. **Code Organization:**
   - Keep API routes modular in `api/app/` subdirectories
   - Use Pydantic schemas for request/response validation
   - Store shared logic in `utils/` modules

2. **Model Development:**
   - Always log experiments to MLflow (hyperparameters, metrics, artifacts)
   - Use feature names consistently across preprocessing and training
   - Validate model on hold-out test set before promotion

3. **Data Management:**
   - Version datasets with DVC; commit `.dvc` files to Git
   - Store credentials in environment variables, never in code
   - Use stratified splits for imbalanced classification tasks

4. **Deployment:**
   - Update container image tags in Helm values (via manifests-cloud-ops)
   - Test API changes locally before pushing
   - Monitor model performance metrics post-deployment

---

## Troubleshooting

### Model Not Updating
- Check MLflow Registry connection: `curl http://mlflow:5000/api/2.0/registered-models/list`
- Verify S3 bucket access: `aws s3 ls s3://arf-ids-model-bucket/`
- Check auto_refresh_worker logs for errors

### Drift Detection False Positives
- Tune ADWIN_DELTA parameter (lower = more sensitive)
- Increase TREND_BUFFER_SIZE for confirming drift
- Check data labeling accuracy in DynamoDB

### Retrain Pipeline Failures
- Verify DynamoDB table has sufficient labeled samples
- Check S3 permissions for dvc push/pull
- Ensure feature consistency between old and new data

---

## References

- **River Documentation:** [https://riverml.xyz/](https://riverml.xyz/) (Adaptive Random Forest, ADWIN)
- **MLflow Documentation:** [https://mlflow.org/](https://mlflow.org/)
- **DVC Documentation:** [https://dvc.org/](https://dvc.org/)
- **Argo Workflows:** [https://argoproj.github.io/workflows/](https://argoproj.github.io/workflows/)
- **Kubernetes:** [https://kubernetes.io/](https://kubernetes.io/)
- **FastAPI:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

---

## License & Support

For questions, issues, or contributions, please contact the development team.

**Last Updated:** 2025


