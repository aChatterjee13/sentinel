# Azure integrations

This document covers the Azure-specific surface added in **workstream #2**:

- Azure Key Vault secret resolution at config-load time (`${azkv:…}`)
- Azure ML model registry backend
- Azure Blob Storage audit shipper
- Azure ML Online Endpoints deployment target
- Azure App Service slot-swap deployment target
- Azure Kubernetes Service (AKS) deployment target
- Azure OpenAI token economics (`azure/…` model prefix)
- `sentinel cloud test` smoke-test CLI

Workstream #2 is the third post-0.1.0 production hardening workstream.
It assumes you are already comfortable with the base config layout in
[`docs/config-reference.md`](config-reference.md) and the hash-chain
audit trail from [`docs/security.md`](security.md).

> **Threat model reminder.** Sentinel assumes a trusted, single-tenant
> deployment behind a corporate identity provider. Every Azure
> integration below authenticates through
> `azure.identity.DefaultAzureCredential`, which chains the
> environment, Managed Identity, Workload Identity, Azure CLI, and
> Azure Developer CLI credential sources in that order. You are
> expected to assign the runtime identity the minimum RBAC roles
> listed per-feature below — the SDK does not grant, rotate, or
> revoke permissions on your behalf.

---

## 0. Install the Azure extra

Every Azure-shaped feature is behind the `[azure]` extra:

```bash
pip install "sentinel-mlops[azure]"
```

This pulls in `azure-identity`, `azure-keyvault-secrets`,
`azure-storage-blob`, `azure-ai-ml`, and `azure-mgmt-web`. The core
SDK (`pip install sentinel-mlops`) still works without any Azure
dependencies installed — every Azure feature is **lazy-imported** so
`import sentinel` stays free of cloud SDKs even when the extra is
present.

AKS additionally requires the Kubernetes client:

```bash
pip install "sentinel-mlops[azure,k8s]"
```

---

## 1. Azure Key Vault secret resolution

### Why

config hardening introduced `SecretStr` wrapping for webhook URLs and basic-auth
passwords, but the values themselves still had to live in environment
variables. For teams that use Azure Key Vault as their source of
truth, that means a brittle shim layer (`export
SLACK_WEBHOOK_URL=$(az keyvault secret show …)`) before every
deploy. cloud integration lets the config file itself reference Key Vault secrets.

### How

Anywhere a string value is accepted in YAML, you can write:

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${azkv:sentinel-prod/slack-webhook}
    - type: pagerduty
      routing_key: ${azkv:sentinel-prod/pagerduty-routing-key}

dashboard:
  server:
    auth: basic
    basic_auth_username: admin
    basic_auth_password: ${azkv:sentinel-prod/dashboard-password}
```

The loader expands `${azkv:vault-name/secret-name}` at the same stage
as the existing `${VAR}` env-var substitution, *before* Pydantic
validation. Each resolved secret is handed to Pydantic as a plain
string, which `SecretStr`-wraps it at the field boundary. The
plaintext never appears in `sentinel config show`, logs, or error
messages — exactly like a config hardening env-var secret would.

### Vault name and secret name rules

The regex that matches `${azkv:…}` tokens is deliberately strict:

- **Vault name**: 3–24 characters, alphanumeric plus hyphens, must
  start and end with an alphanumeric character (Azure Key Vault
  naming rules).
- **Secret name**: 1+ characters, alphanumeric plus hyphens (Azure
  Key Vault naming rules).

Invalid tokens are left as literals and will fail Pydantic validation
downstream. You can use `sentinel config validate --strict` to surface
unresolved `${azkv:…}` tokens at config-load time with the full JSON
path to the offending field.

### Required RBAC

The runtime identity (local dev via `az login`, CI via Workload
Identity, Managed Identity in production) needs the **Key Vault
Secrets User** role on the vault. That single role grants `get` and
`list` on secrets and is the minimum permission required to resolve
`${azkv:…}` tokens.

```bash
az role assignment create \
  --assignee "$RUNTIME_SP_OBJECT_ID" \
  --role "Key Vault Secrets User" \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.KeyVault/vaults/sentinel-prod"
```

### Caching

A module-level `SecretClient` cache keyed by vault URL means a
config that references ten secrets from the same vault only
negotiates one access token. A second lookup of the same secret
(within the same process) returns the cached plaintext.
`sentinel.config.keyvault.clear_cache()` drops the cache — tests call
it between runs.

### Strict vs lenient mode

| Mode | Behaviour on resolution failure |
|---|---|
| Lenient (default) | Literal `${azkv:…}` token is preserved. Usually fails later when Pydantic or the channel implementation tries to parse it. |
| Strict (`--strict`) | Raises `ConfigKeyVaultError` with the vault, secret, and underlying SDK error. |

This mirrors the env-var substitution behaviour. Use strict in CI,
lenient in local dev.

### Versioning is out of scope

Only the **latest** version of a secret is fetched. Pinning to a
specific version via `${azkv:vault/secret/version}` is intentionally
out of scope for cloud integration — rotation is a future workstream concern.

---

## 2. Azure ML model registry backend

### Why

The SDK ships with three registry backends: local filesystem,
Azure ML, and MLflow. Before cloud integration only the local backend was ever
constructed — `SentinelClient.__init__` hard-coded `ModelRegistry()`.
cloud integration wires every backend through the new `RegistryConfig` schema.

### Config

```yaml
registry:
  backend: azure_ml            # local | azure_ml | mlflow
  subscription_id: ${AZURE_SUBSCRIPTION_ID}
  resource_group: sentinel-prod-rg
  workspace_name: sentinel-prod-ws
```

| Field | Required for backend | Description |
|---|---|---|
| `backend` | always | One of `local`, `azure_ml`, `mlflow`. |
| `path` | `local` | Local filesystem path. Default `./registry`. |
| `subscription_id` | `azure_ml` | Azure subscription ID hosting the workspace. |
| `resource_group` | `azure_ml` | Resource group containing the workspace. |
| `workspace_name` | `azure_ml` | Azure ML workspace name. |
| `tracking_uri` | `mlflow` | MLflow tracking URI. Falls back to `MLFLOW_TRACKING_URI`. |

A cross-field validator enforces that `backend: azure_ml` has every
Azure field set and that `backend: mlflow` has a tracking URI
available. Clear errors point at the missing field.

### Required RBAC

The runtime identity needs the **AzureML Data Scientist** role (or
a custom role with `Microsoft.MachineLearningServices/workspaces/
models/*` and `Microsoft.MachineLearningServices/workspaces/
modelversions/*`) on the workspace.

```bash
az role assignment create \
  --assignee "$RUNTIME_SP_OBJECT_ID" \
  --role "AzureML Data Scientist" \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.MachineLearningServices/workspaces/sentinel-prod-ws"
```

### What gets called

- `ModelRegistry.save_model(...)` →
  `ml_client.models.create_or_update(...)`
- `ModelRegistry.load_model(name, version)` →
  `ml_client.models.download(name=…, version=…)`
- `ModelRegistry.list_models()` →
  `ml_client.models.list()`
- `ModelRegistry.delete_model(name, version)` →
  `ml_client.models.archive(name=…, version=…)`

Authentication is always `DefaultAzureCredential`. The `MLClient`
instance is cached on the backend, so repeated calls within the same
process reuse a single authenticated client.

### Smoke test

```bash
sentinel cloud test --config sentinel.yaml --only registry
# [OK] registry     azure_ml backend, 17 model(s) (241ms)
```

---

## 3. Azure Blob Storage audit shipper

### Why

The security hardening audit trail is local-first: every event is written to the
filesystem under `audit/<date>.jsonl` with an HMAC-SHA256 hash chain.
That is the right primitive for tamper-evidence but is useless if the
host disk disappears. cloud integration adds a **shipper** layer that rotates
completed daily files to Azure Blob Storage without ever touching the
hash chain.

### Critical invariant

> **The hash chain is never re-computed on the shipper side.**
> Shipping is strictly downstream of the local chained write. The
> shipper uploads the exact bytes that were HMAC'd on disk. This
> means you can use `sentinel audit verify` against the local copy,
> the blob copy, or any later re-download without breaking the
> chain. security hardening's integrity guarantee is untouched.

### Config

```yaml
audit:
  storage: azure_blob
  path: ./audit/                           # local staging directory
  retention_days: 2555                     # 7 years for FCA compliance
  log_predictions: true
  tamper_evidence: true                    # security hardening hash chain
  signing_key_env: SENTINEL_AUDIT_KEY
  azure_blob:
    account_url: https://sentinelauditprod.blob.core.windows.net
    container: audit
    prefix: claims_fraud_v2/
    delete_local_after_ship: false         # keep local copy by default
```

| Field | Default | Description |
|---|---|---|
| `account_url` | — | Required. The blob storage account URL. |
| `container` | — | Required. Container that will hold the daily files. |
| `prefix` | `""` | Optional blob name prefix (typically `model_name/`). |
| `delete_local_after_ship` | `false` | Delete the local file after the upload returns a successful receipt. Leave `false` unless local disk is a genuine concern — keeping a local copy lets `sentinel audit verify` run without a network round-trip. |

### What gets called

- On day rotation (when the first event of a new day is written),
  `AuditTrail` calls `shipper.ship(previous_day_file)` in
  fire-and-forget mode on a bounded background queue.
- The `AzureBlobShipper` subclass of `ThreadedShipper` drains the
  queue on a worker thread and calls
  `BlobServiceClient.upload_blob(...)`.
- On `AuditTrail.close()` (or process shutdown), the worker thread
  drains the queue deterministically.
- `enforce_retention(cutoff)` calls the shipper's
  `enforce_retention` hook which lists blobs older than the cutoff
  and deletes them.

### Required RBAC

The runtime identity needs the **Storage Blob Data Contributor**
role on the container:

```bash
az role assignment create \
  --assignee "$RUNTIME_SP_OBJECT_ID" \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/sentinelauditprod/blobServices/default/containers/audit"
```

Use **Storage Blob Data Reader** for read-only access (e.g., an
external auditor verifying the chain).

### Back-pressure

The background queue is bounded (default 1000 items). If the queue is
full, the shipper logs a warning and drops the enqueue attempt —
this is **intentional**: the local chained write has already
succeeded, so data is not lost, only delayed until the next
`close()` / retention run catches it up. Production-grade durable
queueing is a operational observability concern.

### Smoke test

```bash
sentinel cloud test --config sentinel.yaml --only audit
# [OK] audit        azure_blob shipper healthy (312ms)
```

---

## 4. Azure ML Online Endpoints deployment target

### Why

Sentinel's deployment strategies (`canary`, `blue_green`, `shadow`)
knew *what* they wanted to do — "send 25% of traffic to v2" — but had
no way to actually do it. cloud integration introduces `BaseDeploymentTarget` and
three Azure implementations. The Azure ML target is the most
strategy-compatible of the three because Azure ML Online Endpoints
natively support weighted traffic split.

### Config

```yaml
deployment:
  strategy: canary                  # canary works cleanly with endpoint traffic split
  target: azure_ml_endpoint
  canary:
    ramp_steps: [5, 25, 50, 100]
    ramp_interval: 30m
  azure_ml_endpoint:
    endpoint_name: claims-fraud-endpoint
    subscription_id: ${AZURE_SUBSCRIPTION_ID}
    resource_group: sentinel-prod-rg
    workspace_name: sentinel-prod-ws
    deployment_name_pattern: "{model_name}-{version}"
```

| Field | Default | Description |
|---|---|---|
| `endpoint_name` | — | Existing Online Endpoint inside the workspace. |
| `subscription_id` | — | Azure subscription ID. |
| `resource_group` | — | Resource group containing the workspace. |
| `workspace_name` | — | Azure ML workspace name. |
| `deployment_name_pattern` | `"{model_name}-{version}"` | Template for computing the deployment name from a model version. |

### What gets called

- `set_traffic_split(model_name, weights)` →
  fetches the endpoint, sets `endpoint.traffic = {deployment: weight,
  …}`, calls
  `ml_client.online_endpoints.begin_create_or_update(endpoint)`, and
  waits for the long-running operation to complete.
- `health_check(model_name, version)` →
  `ml_client.online_deployments.get(…).provisioning_state == "Succeeded"`.
- `rollback_to(model_name, version)` → set traffic fully (100%) to
  the target deployment name.
- `describe(model_name)` → returns `{endpoint, scoring_uri, deployments}`
  for the `sentinel cloud test` CLI.

### Required RBAC

`AzureML Data Scientist` is insufficient — traffic updates need
write access to the endpoint resource. Assign **AzureML Compute
Operator** (or a custom role with
`Microsoft.MachineLearningServices/workspaces/onlineEndpoints/write`
and `…/onlineDeployments/write`) on the workspace.

### Strategy compatibility

| Strategy | Azure ML endpoint | App Service | AKS |
|---|---|---|---|
| `canary` | ✅ | ❌ (slot swap is binary) | ✅ (replica granularity) |
| `blue_green` | ✅ | ✅ | ✅ |
| `shadow` | ✅ (no traffic change) | ✅ | ✅ |
| `direct` | ✅ | ✅ | ✅ |

The incompatible pair `canary` × `azure_app_service` is rejected at
config-load time with a clear error message pointing at both fields.

---

## 5. Azure App Service deployment target

### Why

Many enterprise Sentinel users deploy model-serving FastAPI apps to
Azure App Service with a **staging** deployment slot. The canonical
blue-green pattern is "deploy to staging, health-check, swap slots".
cloud integration makes that a first-class deployment target.

### Config

```yaml
deployment:
  strategy: blue_green              # only blue_green is supported for App Service
  target: azure_app_service
  blue_green:
    warmup_seconds: 60
  azure_app_service:
    subscription_id: ${AZURE_SUBSCRIPTION_ID}
    resource_group: sentinel-prod-rg
    site_name: claims-fraud-api
    production_slot: production
    staging_slot: staging
    health_check_path: /healthz
```

| Field | Default | Description |
|---|---|---|
| `subscription_id` | — | Azure subscription ID. |
| `resource_group` | — | Resource group hosting the App Service. |
| `site_name` | — | App Service name. |
| `production_slot` | `"production"` | Production slot name (rarely changed). |
| `staging_slot` | `"staging"` | Staging slot name. |
| `health_check_path` | `"/healthz"` | Path to GET for health checks. |

### What gets called

- `set_traffic_split({"production": 100, "staging": 0})` →
  `web_apps.begin_swap_slot(source_slot=staging, target_slot=production)`
  and wait for completion.
- `health_check(...)` → `httpx.get(f"https://{site}-{slot}.azurewebsites.net{health_check_path}")`
  and assert 200 OK.
- `rollback_to(...)` → inverse swap (swap production and staging
  back).
- `describe(model_name)` → `{site_url, production_slot, staging_slot,
  slot_hostnames}` for the cloud test CLI.

### Canary is intentionally rejected

App Service slot traffic routing is brittle (`azure_app_service`
honours `x-ms-routing-name` cookies and can assign a percentage, but
behaviour is inconsistent across plans and regions). cloud integration takes the
conservative stance and rejects `strategy: canary` + `target:
azure_app_service` at config validation time with a clear message
pointing at both fields. Use `blue_green` with App Service, or
`canary` with `azure_ml_endpoint` or `aks`.

### Required RBAC

The runtime identity needs the **Website Contributor** role on the
App Service resource (or a custom role with
`Microsoft.Web/sites/slots/slotsswap/Action`):

```bash
az role assignment create \
  --assignee "$RUNTIME_SP_OBJECT_ID" \
  --role "Website Contributor" \
  --scope "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Web/sites/claims-fraud-api"
```

---

## 6. Azure Kubernetes Service (AKS) deployment target

### Why

For Sentinel users running on AKS, the native deployment primitive is
a Kubernetes Deployment + Service, and traffic split is done by
scaling replicas across Deployments. cloud integration ships a target that works
with vanilla K8s (no service-mesh dependency) so any AKS cluster
can host Sentinel-managed canary rollouts out of the box.

### Config

```yaml
deployment:
  strategy: canary
  target: aks
  canary:
    ramp_steps: [10, 30, 60, 100]
    ramp_interval: 15m
  aks:
    namespace: ml-prod
    service_name: claims-fraud
    deployment_name_pattern: "claims-fraud-{version}"
    replicas_total: 10
    kubeconfig_path: ~/.kube/config    # optional
```

| Field | Default | Description |
|---|---|---|
| `namespace` | — | Kubernetes namespace for the Deployments and Service. |
| `service_name` | — | Existing Kubernetes Service that fronts both versions. |
| `deployment_name_pattern` | `"{model_name}-{version}"` | Template for computing the Deployment name from a model version. |
| `replicas_total` | `10` | Total replica count shared between versions. Higher is finer canary granularity. |
| `kubeconfig_path` | `null` | Optional kubeconfig path. When unset, the client tries in-cluster config first, then `~/.kube/config`. |

### What gets called

- `set_traffic_split(model_name, weights)` → for each version in
  `weights`, scale the matching Deployment to
  `round(replicas_total * weight / 100)` replicas. Uses a
  largest-remainder method so the replicas always sum exactly to
  `replicas_total`.
- `health_check(model_name, version)` → checks
  `apps_v1.read_namespaced_deployment_status(...).status.available_replicas
  >= 1`.
- `rollback_to(model_name, version)` → scale the target version to
  `replicas_total` and every other known version to 0.
- `describe(model_name)` → returns the current replica count per
  Deployment in the namespace.

### Replica granularity caveat

The traffic split granularity is **bounded by
`replicas_total`**. With `replicas_total: 10` the minimum non-zero
canary is 10%, so a `ramp_steps: [5, 25, …]` config will round 5%
down to 0 replicas and effectively skip the first step. Either set
`replicas_total: 20` (5% granularity) or adjust `ramp_steps` to
match. `sentinel cloud test --only deploy` flags this by printing
the rounded replica counts.

### Required RBAC

The runtime identity needs a Kubernetes `Role` or `ClusterRole` with
`get`, `list`, `patch`, and `scale` on `apps/deployments` in the
target namespace. For AKS Azure AD integration, bind the role to the
identity via a `RoleBinding`:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: ml-prod
  name: sentinel-deployer
rules:
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "patch"]
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["get", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: ml-prod
  name: sentinel-deployer
subjects:
  - kind: User
    name: "aks-sentinel-runtime@example.com"   # Azure AD identity
roleRef:
  kind: Role
  name: sentinel-deployer
  apiGroup: rbac.authorization.k8s.io
```

---

## 7. Azure OpenAI token economics

### Why

Azure OpenAI is the default LLM provider for BFSI customers who need
data residency and regulatory compliance. Before cloud integration the pricing
table in `sentinel/config/defaults.py` only covered OpenAI and
Anthropic directly — an Azure OpenAI deployment named
`gpt-4o-claims-fraud-prod` would fall through to `default: 0.0` and
every call would look free.

### The `azure/…` prefix convention

cloud integration introduces a simple convention: when recording a token usage
event, prefix the Azure OpenAI deployment name with `azure/`:

```python
tracker.record(
    model="azure/gpt-4o",     # <-- the prefix matters
    input_tokens=1000,
    output_tokens=500,
)
```

The `azure/` prefix buys two things:

1. **Pricing lookup.** `DEFAULT_PRICING` now has
   `azure/gpt-4o`, `azure/gpt-4o-mini`, `azure/gpt-4-turbo`,
   `azure/gpt-35-turbo`, `azure/text-embedding-3-small`, and
   `azure/text-embedding-3-large` entries with current Azure
   OpenAI list prices.
2. **Provider tagging.** The new `provider_from_model()` helper
   classifies the model into `"azure"`, `"openai"`, `"anthropic"`,
   or `"unknown"`. `TokenTracker.record` always adds a
   `provider:<value>` row to the aggregated totals, so dashboards
   can break down cost by provider without any extra config.

### Custom pricing still wins

Azure OpenAI list prices vary by region and commitment tier. If
your enterprise agreement gives you a different per-1K price,
override the defaults in `sentinel.yaml`:

```yaml
llmops:
  token_economics:
    pricing:
      azure/gpt-4o:
        input: 0.0025      # our EA rate
        output: 0.0075
```

User-supplied prices override the defaults; the per-provider tagging
still happens automatically.

### Not in scope for cloud integration

cloud integration does **not** rewrite `LLMOpsClient.log_call` to know about
provider routing, SDK client selection, or Azure-specific auth.
Those remain caller concerns — the token tracker only cares about
the model name and the token counts. Full provider abstraction is a
future workstream concern.

---

## 8. `sentinel cloud test` — smoke-test CLI

### Why

Every Azure feature above fails in subtly different ways when
something is wrong with the environment: missing RBAC, wrong
subscription, expired access token, typo'd resource name, Key Vault
soft-delete active, wrong cluster in kubeconfig. cloud integration adds a single
CLI that pings every configured backend, reports pass/fail with
elapsed time, and exits non-zero if any probe fails.

### Usage

```bash
sentinel cloud test --config sentinel.yaml
# [OK] keyvault     config loaded, all ${azkv:..} resolved (182ms)
# [OK] registry     azure_ml backend, 17 model(s) (241ms)
# [OK] audit        azure_blob shipper healthy (312ms)
# [OK] deploy       azure_ml_endpoint target reachable (198ms)
#
# All 4 backend(s) OK.
```

Each probe runs sequentially and prints:

- `[OK]` with elapsed ms on success
- `[FAIL]` with the exception type and message on failure

A failure in any probe exits with code 1; all four must pass for an
exit code of 0. Probes after a failure still run so you get a full
diagnostic in one command.

### Scoping with `--only`

```bash
sentinel cloud test --only keyvault
sentinel cloud test --only registry
sentinel cloud test --only audit
sentinel cloud test --only deploy
```

Only the selected backend is probed. `--only` accepts exactly one
value and is click-validated against the list
(`keyvault`, `registry`, `audit`, `deploy`). Invalid values are
rejected with exit code 2 and a click usage message.

### What each probe actually does

| Probe | Implementation |
|---|---|
| `keyvault` | `load_config(config_path)` — exercises the loader, including all `${azkv:…}` substitution. Any resolution failure surfaces as `ConfigKeyVaultError`. |
| `registry` | `SentinelClient._build_registry_backend(config)` + `backend.list_models()`. For `local`, this reads the filesystem; for `azure_ml`, it authenticates and calls `ml_client.models.list()`. |
| `audit` | `SentinelClient._build_audit_shipper(config)` + `shipper.health_check()` (when the shipper exposes one) + `shipper.close()` to drain the worker thread. For `local`, no remote probe runs. |
| `deploy` | `DeploymentManager._build_target(config.deployment)` + `target.describe(model.name)`. For Azure targets, this authenticates and fetches the endpoint / site / namespace descriptor. |

### Troubleshooting table

Every failure mode below is wrapped in an `[FAIL]` line with a
descriptive message. The table maps the common ones to root causes.

| Failure message | Likely cause | Fix |
|---|---|---|
| `[FAIL] keyvault ConfigKeyVaultError: could not construct SecretClient for vault 'X'` | The `azure` extra is not installed. | `pip install "sentinel-mlops[azure]"` |
| `[FAIL] keyvault ConfigKeyVaultError: ... DefaultAzureCredential failed to retrieve a token` | No identity source available. | `az login` locally, or assign Managed Identity / Workload Identity in production. |
| `[FAIL] keyvault ConfigKeyVaultError: ... Forbidden ... does not have secrets get permission` | Missing `Key Vault Secrets User` role. | Run the `az role assignment create` snippet from §1. |
| `[FAIL] registry ResourceNotFoundError: Workspace 'X' was not found` | Wrong subscription, resource group, or workspace name. | Check the three fields in `registry:` against `az ml workspace show`. |
| `[FAIL] registry HttpResponseError: Authentication failed` | `DefaultAzureCredential` resolved but the identity lacks `AzureML Data Scientist`. | Run the role assignment snippet from §2. |
| `[FAIL] audit ResourceNotFoundError: The specified container does not exist` | `audit.azure_blob.container` typo or container not yet provisioned. | Create the container: `az storage container create --name audit --account-name sentinelauditprod --auth-mode login`. |
| `[FAIL] audit ClientAuthenticationError` | Missing `Storage Blob Data Contributor` role. | Run the role assignment snippet from §3. |
| `[FAIL] deploy ResourceNotFoundError: Endpoint 'X' was not found` | `azure_ml_endpoint.endpoint_name` typo or endpoint not provisioned yet. | Create the endpoint first with `az ml online-endpoint create`, then re-run. |
| `[FAIL] deploy HttpResponseError: 403 ... does not have permission to perform action 'Microsoft.Web/sites/slots/swap'` | Missing `Website Contributor` role on the App Service. | Run the role assignment snippet from §5. |
| `[FAIL] deploy ApiException: (401) Reason: Unauthorized` (AKS) | `kubeconfig_path` points at a cluster your identity cannot talk to, or the Azure AD token for the cluster has expired. | `kubectl get deploy -n <namespace>` to confirm access, then re-run. |

### Recommended workflow

1. Run `sentinel cloud test` in CI on every merge to `main`, gated
   by an `[azure-integration]` job label so it only runs when your
   runner has Azure credentials. This catches RBAC drift before
   production.
2. Run it once manually from every new environment
   (dev / staging / prod) before onboarding it to the auto-deploy
   pipeline. The four OK lines are the signal that every backend
   is reachable and every role assignment is in place.
3. Include `sentinel cloud test` in your oncall runbook as the
   first diagnostic step when anything Azure-shaped breaks.

---

## 9. What's intentionally out of scope for cloud integration

- **Azure OpenAI as a provider abstraction in `LLMOpsClient`.** cloud integration
  adds pricing and provider tagging, but does not rewrite the log
  path to know about providers. Deferred to future workstream.
- **Live subscription integration tests in CI.** The unit test
  suite uses mocks. Tape-replay tests (vcrpy) are a deferred nice
  -to-have — the plan was to record them once against a real Azure
  subscription and commit the tapes, but this shipped without them.
- **Azure AI Foundry and Azure ML Prompt Flow integrations.**
  Deferred.
- **SageMaker registry backend and deploy target.** cloud integration is
  Azure-focused. AWS stays at the existing helper-shipper level.
- **Key Vault secret versioning.** `${azkv:…}` resolves to the
  latest version only. Rotation is a future workstream concern.
- **Azure Monitor / Managed Grafana metric export.** That belongs
  in operational observability (operational observability).
- **Durable audit shipper queue.** The in-process bounded queue is
  intentionally simple. Production-grade queueing is a operational observability
  concern.

## 10. Further reading

- [`docs/config-reference.md`](config-reference.md) — full field
  reference, including the new `registry`, `audit.azure_blob`, and
  per-target deployment sub-configs.
- [`docs/security.md`](security.md) audit hash chain, which
  every cloud integration shipper runs strictly downstream of.
- [`CHANGELOG.md`](../CHANGELOG.md) entry with the full list
  of new modules, schema changes, and tests.
