- `cloud-ops` — chỉ chứa code, container build, ML code, scripts và workflows liên quan đến build/test.
- `manifests-cloud-ops` — chứa Helm charts, `values/`, `apps/` và các manifest khác (deployments, services, ingress, ...).

Các thư mục quan trọng trong repository này:

- `api/` — mã nguồn server (FastAPI/Flask/... tùy file nội dung).
- `mlflow/` — cấu hình hoặc image liên quan tới MLflow (nếu có).
- `config/` — mẫu cấu hình, biến môi trường (ví dụ `env.example`).
- `workflows/` — workflow/automation (K8s workflows/Argo/k8s pipeline) — có thể giữ hoặc chuyển sang nơi khác tuỳ cần.
- `.github/workflows/` — GitHub Actions workflows (ví dụ `build-api.yml` để build & push Docker image).

CI/CD hiện tại (tóm tắt)

- `build-api.yml` (trong `cloud-ops`) chịu trách nhiệm:
  - Build và tag image (ví dụ `bqmxnh/arf-ids-api:YYYYMMDDHHMMSS`).
  - Push image lên Docker registry.
  - Trigger workflow ở repo `manifests-cloud-ops` để update Helm values (image tag).

- `update-helm.yml` (sẽ nằm trong `manifests-cloud-ops`) chịu trách nhiệm:
  - Nhận trigger (via `workflow_dispatch` hoặc `repository_dispatch`).
  - Cập nhật `values/api/values.yaml` với tag mới.
  - Commit & push thay đổi trở lại `manifests-cloud-ops`.


