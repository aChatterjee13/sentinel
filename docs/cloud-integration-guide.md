# Cloud Integration Guide

This guide covers Sentinel's multi-cloud integration surface — model
registry, audit trail, deployment targets, secret resolution,
notifications, and diagnostics — across **Azure**, **AWS**, and **GCP**.

Sentinel is **Azure-first** (see [`docs/azure.md`](azure.md) for the
full Azure surface), but its architecture is deliberately
cloud-agnostic. Every cloud-touching feature is isolated behind a
small abstract interface. Cloud SDKs are lazy-imported inside concrete
implementations so that `import sentinel` never pulls heavyweight
packages.

> **Prerequisites.** Familiarity with the base config layout
> ([`docs/config-reference.md`](config-reference.md)) and the audit
> trail security model ([`docs/security.md`](security.md)).

---

## Implementation Status

All backends in the table below are **shipped and tested**. Each has a
concrete implementation in `sentinel/foundation/`, `sentinel/action/`,
or `sentinel/config/`. Install the matching extra to activate.

| Capability | Azure | AWS | GCP | Databricks |
|---|---|---|---|---|
| **Secret resolution** | ✅ Key Vault (`${azkv:…}`) | ✅ Secrets Manager (`${awssm:…}`) | ✅ Secret Manager (`${gcpsm:…}`) | — |
| **Model registry** | ✅ Azure ML (`azure_ml`) | ✅ SageMaker (`sagemaker`) | ✅ Vertex AI (`vertex_ai`) | ✅ Unity Catalog (`databricks`) |
| **Audit shipper** | ✅ Blob Storage (`azure_blob`) | ✅ S3 (`s3`) | ✅ GCS (`gcs`) | — |
| **Deployment target** | ✅ ML Endpoint, App Service, AKS | ✅ SageMaker Endpoint (`sagemaker_endpoint`) | ✅ Vertex AI Endpoint (`vertex_ai_endpoint`) | — |
| **Notification** | ✅ Teams | ✅ (via generic webhook) | ✅ (via generic webhook) | — |
| **Pipeline runner** | ✅ Azure ML Pipelines | — (planned) | — (planned) | — |

### Install extras

```bash
pip install "sentinel-mlops[azure]"     # Azure Key Vault, ML, Blob Storage
pip install "sentinel-mlops[aws]"       # boto3, SageMaker
pip install "sentinel-mlops[gcp]"       # google-cloud-storage, Vertex AI
pip install "sentinel-mlops[databricks]" # databricks-sdk
```

---

## Overview

### Cloud-agnostic architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SentinelClient                            │
│  from_config("sentinel.yaml")                               │
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌────────────┐  ┌────────┐  │
│  │  Drift    │  │  Alerts   │  │ Deployment │  │ Audit  │  │
│  │  Engine   │  │  Engine   │  │  Manager   │  │ Trail  │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬──────┘  └───┬────┘  │
└────────┼──────────────┼──────────────┼──────────────┼───────┘
         │              │              │              │
    ┌────▼────┐   ┌─────▼────┐  ┌─────▼──────┐ ┌────▼──────┐
    │Registry │   │ Channel  │  │ Deployment │ │  Audit    │
    │ Backend │   │  (ABC)   │  │  Target    │ │  Shipper  │
    │  (ABC)  │   │          │  │   (ABC)    │ │   (ABC)   │
    └────┬────┘   └────┬─────┘  └─────┬──────┘ └────┬──────┘
         │             │              │              │
  ┌──────┼──────┐  ┌───┼───┐   ┌─────┼──────┐  ┌───┼────────┐
  │      │      │  │   │   │   │     │      │  │   │        │
  ▼      ▼      ▼  ▼   ▼   ▼   ▼     ▼      ▼  ▼   ▼        ▼
Local  Azure  SM  Sl  Te  PD  Local Azure  SM  Null Azure  S3
       ML        ack ams       ML    EP        Blob
  Vrtx  MLfl  DB        WH         AKS  Vrtx          GCS
              |                     |    EP
         SageMaker             SageMaker
                               Endpoint

SM = SageMaker    DB = Databricks    WH = Webhook    PD = PagerDuty
Vrtx = Vertex AI  MLfl = MLflow      EP = Endpoint
```

### Extension points summary

| Extension point | ABC | Location | Shipped backends |
|---|---|---|---|
| Model registry | `BaseRegistryBackend` | `sentinel/foundation/registry/backends/base.py` | `local`, `azure_ml`, `mlflow`, `sagemaker`, `vertex_ai`, `databricks` |
| Audit shipper | `BaseAuditShipper` | `sentinel/foundation/audit/shipper.py` | `NullShipper`, `AzureBlobShipper`, `S3Shipper`, `GcsShipper` |
| Deployment target | `BaseDeploymentTarget` | `sentinel/action/deployment/targets/base.py` | `local`, `azure_ml_endpoint`, `azure_app_service`, `aks`, `sagemaker_endpoint`, `vertex_ai_endpoint` |
| Notification channel | `BaseChannel` | `sentinel/action/notifications/channels/base.py` | `slack`, `teams`, `pagerduty`, `email`, `webhook` |
| Secret resolution | Pattern-matched substitution | `sentinel/config/` | `${azkv:…}` (Key Vault), `${awssm:…}` (Secrets Manager), `${gcpsm:…}` (Secret Manager) |
| Pipeline runner | Callable protocol | `sentinel/integrations/azure/pipeline_runner.py` | `AzureMLPipelineRunner` |

### How backends are resolved from YAML

When `SentinelClient.from_config()` loads a YAML file, it:

1. Substitutes `${VAR}` environment variables and cloud secret tokens
   (`${azkv:…}`, `${awssm:…}`, `${gcpsm:…}`).
2. Validates against Pydantic schemas in `sentinel/config/schema.py`.
3. Calls factory methods to construct the correct backend:
   - `SentinelClient._build_registry_backend(config)` reads
     `config.registry.backend` and returns the matching
     `BaseRegistryBackend`.
   - `SentinelClient._build_audit_shipper(config)` reads
     `config.audit.storage` and returns the matching
     `BaseAuditShipper`.
   - `DeploymentManager._build_target(config.deployment)` reads
     `config.deployment.target` and returns the matching
     `BaseDeploymentTarget`.

Adding a new cloud backend means: (1) implement the ABC, (2) add a
config sub-model, (3) add a branch to the factory, and (4) register
the optional dependency in `pyproject.toml`.

---

## AWS Integration

### Install the AWS extra

```bash
pip install "sentinel-mlops[aws]"
```

This pulls in `boto3>=1.28` and `sagemaker>=2.200`. Like all Sentinel
extras, these are lazy-imported — the core SDK works without them.

---

### Model Registry → Amazon SageMaker Model Registry

SageMaker Model Registry stores model packages with versioning,
approval status, and metadata. Sentinel's `SageMakerRegistryBackend`
(in `sentinel/foundation/registry/backends/sagemaker.py`) maps:

- `model_name` → SageMaker Model Package Group
- `version` → Model Package within that group

#### Config

```yaml
registry:
  backend: sagemaker
  region_name: us-east-1
  role_arn: arn:aws:iam::123456789012:role/SentinelSageMakerRole
```

#### Required IAM permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModelPackageGroup",
        "sagemaker:DescribeModelPackageGroup",
        "sagemaker:ListModelPackageGroups",
        "sagemaker:CreateModelPackage",
        "sagemaker:DescribeModelPackage",
        "sagemaker:ListModelPackages",
        "sagemaker:DeleteModelPackage",
        "sagemaker:UpdateModelPackage"
      ],
      "Resource": "arn:aws:sagemaker:*:*:model-package-group/sentinel-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::sentinel-artifacts/*"
    }
  ]
}
```

---

### Audit Trail → Amazon S3

Sentinel ships with `S3Shipper` in
`sentinel/integrations/aws/s3_audit.py`.

`S3Shipper` extends `ThreadedShipper` — the same base class used by
`AzureBlobShipper` and `GcsShipper`. The architecture is:

1. `AuditTrail` writes events to local JSON-Lines files (one per day).
2. On day rotation, it calls `shipper.ship(previous_day_file)`.
3. `ThreadedShipper` enqueues the file on a bounded background queue.
4. A worker thread calls `_ship_sync()`, which uploads via boto3 S3.
5. Retries (exponential backoff, max 3 attempts) are handled
   automatically. Failures are logged, never raised — the hot write
   path is never blocked.

The shipper uses `boto3.client("s3")` which resolves credentials
through the standard AWS credential chain (environment variables →
`~/.aws/credentials` → instance profile → ECS task role).

#### Config

```yaml
audit:
  storage: s3
  path: ./audit/
  retention_days: 2555
  log_predictions: true
  s3:
    bucket: my-sentinel-audit
    prefix: audit-logs/
    region: us-east-1
    delete_local_after_ship: false
```

#### Required IAM permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:HeadBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-sentinel-audit",
        "arn:aws:s3:::my-sentinel-audit/*"
      ]
    }
  ]
}
```

> **Tip — immutable storage.** Enable S3 Object Lock (WORM) on the
> audit bucket for compliance. The shipper never deletes remote
> objects, so WORM policies are fully compatible.

---

### Deployment → Amazon SageMaker Endpoints

Sentinel ships with `SageMakerEndpointTarget` in
`sentinel/action/deployment/targets/sagemaker.py`. It maps Sentinel
model versions to SageMaker production variants and uses
`UpdateEndpointWeightsAndCapacities` for zero-downtime traffic shifts.

#### Config

```yaml
deployment:
  strategy: canary
  target: sagemaker_endpoint
  canary:
    initial_traffic_pct: 5
    ramp_steps: [5, 25, 50, 100]
    ramp_interval: 1h
  sagemaker_endpoint:
    endpoint_name: fraud-detector-prod
    region: us-east-1
    role_arn: arn:aws:iam::123456789012:role/SentinelSageMakerRole
    variant_name_pattern: "{model_name}-{version}"
    instance_type: ml.m5.large
    instance_count: 2
```

#### Required IAM permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:DescribeEndpoint",
        "sagemaker:UpdateEndpointWeightsAndCapacities",
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:*:*:endpoint/fraud-detector-*"
    }
  ]
}
```

---

### Secrets → AWS Secrets Manager

Sentinel resolves `${awssm:secret-name}` tokens at config load time,
following the same pattern as Azure Key Vault. The implementation is
in `sentinel/config/aws_secrets.py`.

#### Token syntax

```
${awssm:secret-name}              # default region
${awssm:secret-name:us-east-1}    # explicit region
```

The resolver uses boto3's default credential chain.

#### Config usage

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${awssm:prod/sentinel/slack-webhook:us-east-1}
    - type: pagerduty
      routing_key: ${awssm:prod/sentinel/pagerduty-key}
```

#### Required IAM permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:*:*:secret:prod/sentinel/*"
    }
  ]
}
```

---

### Notifications (AWS)

AWS teams can use the built-in **webhook** channel to post alerts to
any HTTP endpoint, including SNS HTTP subscriptions, Lambda function
URLs, or custom APIs. Dedicated SNS and CloudWatch channels are
planned for a future release.

```yaml
alerts:
  channels:
    - type: webhook
      webhook_url: ${awssm:prod/sentinel/alert-endpoint-url}
      headers:
        Content-Type: application/json
```

---

## GCP Integration

### Install the GCP extra

```bash
pip install "sentinel-mlops[gcp]"
```

This pulls in `google-cloud-storage>=2.10` and `google-auth>=2.25`.
For Vertex AI features (registry and deployment targets), also install:

```bash
pip install google-cloud-aiplatform>=1.40
```

GCP authentication uses Application Default Credentials (ADC) —
`gcloud auth application-default login` for local development, or a
service account key / Workload Identity for production.

---

### Model Registry → Vertex AI Model Registry

Sentinel ships `VertexAIRegistryBackend` in
`sentinel/foundation/registry/backends/vertex_ai.py`. It maps:

- `model_name` → Vertex AI Model resource (by `display_name`)
- `version` → Vertex AI Model Version (via `version_aliases`)

#### Config

```yaml
registry:
  backend: vertex_ai
  project: my-gcp-project
  location: us-central1
```

#### Required GCP IAM roles

| Role | Why |
|------|-----|
| `roles/aiplatform.user` | Upload, list, and manage models |
| `roles/storage.objectViewer` | Read model artifacts from GCS |
| `roles/storage.objectCreator` | Upload model artifacts to GCS |

```bash
gcloud projects add-iam-policy-binding my-gcp-project \
    --member="serviceAccount:sentinel@my-gcp-project.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

---

### Audit Trail → Google Cloud Storage

Sentinel ships `GcsShipper` in
`sentinel/integrations/gcp/gcs_audit.py`, following the same
`ThreadedShipper` pattern as Azure Blob and S3 shippers.

#### Config

```yaml
audit:
  storage: gcs
  path: ./audit/
  retention_days: 2555
  gcs:
    bucket: sentinel-audit-prod
    prefix: audit-logs/
    project: my-gcp-project
    delete_local_after_ship: false
```

#### Required GCP IAM roles

| Role | Why |
|------|-----|
| `roles/storage.objectCreator` | Upload audit files |
| `roles/storage.legacyBucketReader` | `health_check()` calls `bucket.reload()` |

> **Tip — retention lock.** Enable a GCS Bucket Lock retention policy
> for compliance. The shipper never deletes remote objects.

---

### Deployment → Vertex AI Endpoints

Sentinel ships `VertexAIEndpointTarget` in
`sentinel/action/deployment/targets/vertex_ai.py`. Vertex AI
endpoints support native traffic splitting across deployed models,
making them ideal for canary deployments.

#### Config

```yaml
deployment:
  strategy: canary
  target: vertex_ai_endpoint
  vertex_ai_endpoint:
    endpoint_name: fraud-detector-prod
    project: my-gcp-project
    location: us-central1
    deployed_model_name_pattern: "{model_name}-{version}"
    machine_type: n1-standard-4
```

#### Required GCP IAM roles

| Role | Why |
|------|-----|
| `roles/aiplatform.user` | Deploy models, update traffic splits |
| `roles/aiplatform.serviceAgent` | Runtime serving (service account) |

---

### Secrets → Google Secret Manager

Sentinel resolves `${gcpsm:project-id/secret-name}` tokens at config
load time. The implementation is in `sentinel/config/gcp_secrets.py`.

#### Token syntax

```
${gcpsm:my-gcp-project/my-secret-name}
```

The resolver uses Google Application Default Credentials.

#### Config usage

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${gcpsm:my-gcp-project/sentinel-slack-webhook}
```

#### Required GCP IAM roles

| Role | Why |
|------|-----|
| `roles/secretmanager.secretAccessor` | Read secret values |

```bash
gcloud secrets add-iam-policy-binding sentinel-slack-webhook \
    --member="serviceAccount:sentinel@my-project.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

---

### Notifications (GCP)

GCP teams can use the built-in **webhook** channel to post alerts to
Cloud Functions, Cloud Run, or any HTTP endpoint. A dedicated Pub/Sub
channel is planned for a future release.

```yaml
alerts:
  channels:
    - type: webhook
      webhook_url: ${gcpsm:my-gcp-project/sentinel-alert-endpoint}
      headers:
        Content-Type: application/json
```

---

## Databricks Integration

### Install the Databricks extra

```bash
pip install "sentinel-mlops[databricks]"
```

This pulls in `databricks-sdk`. Authentication uses the Databricks
SDK's default credential chain (token, OAuth, or Azure AAD).

---

### Model Registry → Databricks Unity Catalog

Sentinel ships `DatabricksRegistryBackend` in
`sentinel/foundation/registry/backends/databricks.py`. It maps:

- `model_name` → Unity Catalog Registered Model
  (`catalog.schema.model_name`)
- `version` → Model Version within that registered model

If the model name already contains dots (e.g.
`ml_catalog.production.fraud`), it is used as-is. Otherwise it is
placed under the configured `catalog` and `schema`.

#### Config

```yaml
registry:
  backend: databricks
  host: https://adb-xxx.azuredatabricks.net
  token: ${azkv:my-vault/databricks-token}  # or use DATABRICKS_TOKEN env var
  # Optional: default catalog and schema for unqualified model names
  # catalog: ml_catalog
  # schema: default
```

#### Required Databricks permissions

| Permission | Why |
|------------|-----|
| `USE CATALOG` on the target catalog | Access the catalog |
| `USE SCHEMA` on the target schema | Access the schema |
| `CREATE MODEL` on the schema | Register new models |
| `MANAGE` on registered models | Update, delete model versions |

```sql
GRANT USE CATALOG ON CATALOG ml_catalog TO `sentinel-service-principal`;
GRANT USE SCHEMA ON SCHEMA ml_catalog.production TO `sentinel-service-principal`;
GRANT CREATE MODEL ON SCHEMA ml_catalog.production TO `sentinel-service-principal`;
```

---

## Multi-Cloud Architecture

Sentinel supports running across cloud providers through config
inheritance. A base config defines shared policies, and
environment-specific configs override the cloud backends.

### Config inheritance example

```yaml
# base.yaml — shared across all environments
version: "1.0"
model:
  name: claims_fraud_v2
  type: classification
  framework: xgboost

drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d

alerts:
  policies:
    cooldown: 1h
    escalation:
      - after: 0m
        channels: [slack]
        severity: [medium, high, critical]
```

```yaml
# aws-prod.yaml — AWS production
extends: base.yaml

registry:
  backend: sagemaker
  sagemaker:
    region: us-east-1
    role_arn: ${awssm:prod/sentinel/sagemaker-role}
    model_package_group_prefix: sentinel-prod

audit:
  storage: s3
  s3:
    bucket: sentinel-audit-prod
    prefix: audit-logs/
    region: us-east-1

deployment:
  strategy: canary
  target: sagemaker_endpoint
  sagemaker_endpoint:
    endpoint_name: fraud-detector-prod
    region: us-east-1
    role_arn: ${awssm:prod/sentinel/sagemaker-role}

alerts:
  channels:
    - type: sns
      topic_arn: ${awssm:prod/sentinel/sns-topic-arn}
    - type: slack
      webhook_url: ${awssm:prod/sentinel/slack-webhook}
```

```yaml
# gcp-staging.yaml — GCP staging
extends: base.yaml

registry:
  backend: vertex_ai
  vertex_ai:
    project: sentinel-staging
    location: us-central1

audit:
  storage: gcs
  gcs:
    bucket: sentinel-audit-staging
    project: sentinel-staging

deployment:
  strategy: canary
  target: vertex_ai_endpoint
  vertex_ai_endpoint:
    endpoint_name: fraud-detector-staging
    project: sentinel-staging

alerts:
  channels:
    - type: pubsub
      project: sentinel-staging
      topic: sentinel-alerts
```

### Usage

```bash
# AWS production
sentinel check --config aws-prod.yaml

# GCP staging
sentinel check --config gcp-staging.yaml

# Smoke-test both environments
sentinel cloud test --config aws-prod.yaml
sentinel cloud test --config gcp-staging.yaml
```

---

## Step-by-Step: Adding a Custom Backend

This is the general recipe for adding any new cloud backend. The time
investment is roughly 2–4 hours for a complete, tested implementation.

### 1. Choose the ABC to implement

| I want to add a… | Implement | Key methods |
|---|---|---|
| Model registry backend | `BaseRegistryBackend` | `save`, `load`, `list_versions`, `list_models`, `delete`, `exists` |
| Audit shipper | `ThreadedShipper` | `_ship_sync`, `enforce_retention` |
| Deployment target | `BaseDeploymentTarget` | `set_traffic_split`, `health_check`, `rollback_to` |
| Notification channel | `BaseChannel` | `send` |
| Secret resolver | Pattern function | `substitute_xxx(value, strict) -> str` |

### 2. Create your module

Place it under `sentinel/integrations/{provider}/`:

```
sentinel/integrations/
├── aws/
│   ├── __init__.py
│   ├── s3_audit.py             # existing
│   ├── sagemaker_registry.py   # new
│   ├── sagemaker_endpoint.py   # new
│   ├── sns_channel.py          # new
│   └── ecs_target.py           # new
├── gcp/
│   ├── __init__.py
│   ├── gcs_audit.py            # new
│   ├── vertex_registry.py      # new
│   ├── vertex_endpoint.py      # new
│   └── pubsub_channel.py       # new
└── azure/
    ├── __init__.py
    ├── blob_audit.py           # existing
    └── pipeline_runner.py      # existing
```

### 3. Follow the lazy-import pattern

Every backend must lazy-import its cloud SDK inside `__init__()` so
that `import sentinel` never pulls in heavyweight packages:

```python
class MyBackend(BaseRegistryBackend):
    def __init__(self, **kwargs):
        try:
            import my_cloud_sdk  # type: ignore[import-not-found]
        except ImportError as e:
            raise RegistryError(
                "my-cloud extra not installed — "
                "`pip install sentinel-mlops[my-cloud]`"
            ) from e
        self._client = my_cloud_sdk.Client()
```

### 4. Add a config sub-model

In `sentinel/config/schema.py`:

1. Create a `_Base` sub-model for your backend's config fields.
2. Add the backend name to the relevant `Literal` union.
3. Add a nullable field for the sub-config.
4. Add a `model_validator` branch for validation.

### 5. Wire into the factory

Add a branch to the appropriate factory method:

- **Registry:** `SentinelClient._build_registry_backend()` in
  `sentinel/core/client.py`
- **Audit shipper:** `SentinelClient._build_audit_shipper()` in
  `sentinel/core/client.py`
- **Deployment target:** `DeploymentManager._build_target()` in
  `sentinel/action/deployment/manager.py`
- **Notification channel:** Channel resolver in
  `sentinel/action/notifications/engine.py`

### 6. Add optional dependency to pyproject.toml

```toml
[project.optional-dependencies]
my-cloud = ["my-cloud-sdk>=1.0"]
```

### 7. Write tests

```python
# tests/unit/integrations/aws/test_sagemaker_registry.py

import pytest
from unittest.mock import MagicMock, patch


class TestSageMakerRegistryBackend:
    """Unit tests with mocked boto3 client."""

    @pytest.fixture
    def backend(self):
        with patch("boto3.client") as mock_client:
            from sentinel.integrations.aws.sagemaker_registry import (
                SageMakerRegistryBackend,
            )
            backend = SageMakerRegistryBackend(
                region="us-east-1",
                role_arn="arn:aws:iam::123:role/test",
            )
            backend._sm_client = mock_client.return_value
            yield backend

    def test_save_creates_group_and_package(self, backend):
        backend._sm_client.describe_model_package_group.side_effect = (
            Exception("not found")
        )
        backend._sm_client.create_model_package.return_value = {
            "ModelPackageArn": "arn:test"
        }

        uri = backend.save("fraud", "1.0", {"metrics": {"f1": 0.95}})

        backend._sm_client.create_model_package_group.assert_called_once()
        assert uri == "arn:test"

    def test_exists_returns_false_for_missing(self, backend):
        backend._sm_client.get_paginator.return_value.paginate.return_value = []
        assert backend.exists("fraud", "999") is False
```

For integration tests against real cloud services, use `moto` (AWS)
or `google-cloud-testutils` (GCP):

```python
# tests/integration/aws/test_s3_audit_integration.py

import pytest

pytest.importorskip("moto")

from moto import mock_aws


@mock_aws
class TestS3ShipperIntegration:
    def test_upload_and_health_check(self, tmp_path):
        import boto3

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-audit")

        from sentinel.integrations.aws.s3_audit import S3Shipper

        shipper = S3Shipper(bucket="test-audit", region="us-east-1")
        assert shipper.health_check() is True

        audit_file = tmp_path / "audit-2025-01-15.jsonl"
        audit_file.write_text('{"event": "test"}')
        shipper.ship(audit_file)
        shipper.close()
```

---

## `sentinel cloud test` CLI

The built-in smoke-test command probes every configured cloud backend
with lightweight reachability checks. No records are written, no
models are deployed.

```bash
# Test all backends
sentinel cloud test --config sentinel.yaml

# Test a specific backend
sentinel cloud test --config sentinel.yaml --only registry
sentinel cloud test --config sentinel.yaml --only audit
sentinel cloud test --config sentinel.yaml --only deploy
sentinel cloud test --config sentinel.yaml --only keyvault
```

Example output:

```
[OK] keyvault     config loaded, all secrets resolved (234ms)
[OK] registry     sagemaker backend, 3 model(s) (891ms)
[OK] audit        s3 shipper healthy (342ms)
[OK] deploy       sagemaker_endpoint target {'endpoint': 'fraud-prod', 'status': 'InService'} (567ms)
```

The command exits `0` on success, `1` on any failure. Supports all
three cloud providers and the Databricks backend.

---

## Testing Cloud Integrations

### `sentinel cloud test` CLI (details)

The built-in smoke-test command was described above. Here are
additional patterns for CI/CD:

```bash
# Smoke-test all environments in CI
sentinel cloud test --config aws-prod.yaml
sentinel cloud test --config gcp-staging.yaml
sentinel cloud test --config azure-prod.yaml
```

### Mocking cloud services in unit tests

| Provider | Mocking library | Install |
|----------|----------------|---------|
| AWS | [moto](https://github.com/getmoto/moto) | `pip install moto[all]` |
| GCP | `unittest.mock` + `google-cloud-testutils` | `pip install google-cloud-testutils` |
| Azure | `unittest.mock` (Sentinel's test suite approach) | built-in |

### Integration test patterns

For real cloud integration tests (run in CI with cloud credentials):

```python
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("AWS_DEFAULT_REGION"),
    reason="AWS credentials not configured",
)
class TestSageMakerRegistryIntegration:
    def test_full_lifecycle(self):
        backend = SageMakerRegistryBackend(
            region="us-east-1",
            role_arn=os.environ["SAGEMAKER_ROLE_ARN"],
        )
        # save → load → list → delete cycle
        uri = backend.save("test-model", "1.0", {"metrics": {"f1": 0.9}})
        assert backend.exists("test-model", "1.0")

        data = backend.load("test-model", "1.0")
        assert data["metrics"]["f1"] == 0.9

        backend.delete("test-model", "1.0")
        assert not backend.exists("test-model", "1.0")
```

---

## Quick Reference: All Backends

| Feature | Azure | AWS | GCP | Databricks |
|---------|-------|-----|-----|------------|
| **Model registry** | ✅ Azure ML (`azure_ml`) | ✅ SageMaker (`sagemaker`) | ✅ Vertex AI (`vertex_ai`) | ✅ Unity Catalog (`databricks`) |
| **Audit shipper** | ✅ Azure Blob (`azure_blob`) | ✅ S3 (`s3`) | ✅ GCS (`gcs`) | — |
| **Deployment target** | ✅ ML Endpoint, App Service, AKS | ✅ SageMaker Endpoint | ✅ Vertex AI Endpoint | — |
| **Secret resolution** | ✅ Key Vault (`${azkv:…}`) | ✅ Secrets Manager (`${awssm:…}`) | ✅ Secret Manager (`${gcpsm:…}`) | — |
| **Notifications** | ✅ Teams | ✅ webhook | ✅ webhook | — |
| **Pipeline runner** | ✅ Azure ML Pipelines | — (planned) | — (planned) | — |

✅ = shipped and tested

---

## Appendix: pyproject.toml extras

```toml
[project.optional-dependencies]
azure = [
    "azure-ai-ml>=1.12",
    "azure-storage-blob>=12.19",
    "azure-identity>=1.15",
    "azure-keyvault-secrets>=4.8",
    "azure-mgmt-web>=7.2",
]
k8s = ["kubernetes>=29.0"]
aws = ["boto3>=1.28", "sagemaker>=2.200"]
gcp = [
    "google-cloud-storage>=2.10",
    "google-auth>=2.25",
    "google-cloud-aiplatform>=1.40",
    "google-cloud-secret-manager>=2.18",
]
databricks = ["databricks-sdk>=0.20"]
```
