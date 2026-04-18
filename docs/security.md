# Security

This document covers the security features added in **workstream #3**:

- Tamper-evident audit trails (HMAC hash chain + `sentinel audit verify`)
- RBAC for the dashboard (viewer / operator / admin, custom roles, route guards)
- HTTP Basic + Bearer JWT authentication modes
- CSRF protection (double-submit cookie)
- In-memory token-bucket rate limiting
- Security response headers (HSTS, CSP, frame guards, …)
- Signed configs (HMAC-detached `.sig` sidecar files)

Workstream #3 is the second of six post-0.1.0 production hardening
workstreams. It assumes you have already worked through the
[`docs/quickstart.md`](quickstart.md) flow and understand the basic
config layout in [`docs/config-reference.md`](config-reference.md).

> **Threat model boundaries.** Sentinel is designed for trusted
> single-tenant deployments behind a corporate VPN or identity provider.
> It is **not** a multi-tenant SaaS surface. The hardening below closes
> the gaps that any enterprise security review will flag in the first
> 30 minutes — accidental tampering, missing access control, plaintext
> secrets in logs — but the SDK still assumes the host process, file
> system, and signing keys are under your control.

---

## 1. Tamper-evident audit trail

### Why

The audit trail (`sentinel/foundation/audit/trail.py`) is the
compliance backbone — every drift detection, alert, deployment, and
prediction can be logged here for FCA / EU AI Act review. Without
integrity protection, anyone with write access to the audit directory
could silently alter history and there'd be no way to tell.

### How

When `audit.tamper_evidence: true`, every event is written with two
extra fields:

- `previous_hash` — the `event_hmac` of the previous event in the
  chain (or `null` for the first event in a fresh trail).
- `event_hmac` — HMAC-SHA256 of the canonical JSON of every other
  field, computed with a key from the configured keystore.

Editing, inserting, or deleting any event breaks the chain and is
detected by `sentinel audit verify`.

### Configuration

```yaml
audit:
  storage: local
  path: ./audit/
  retention_days: 2555
  log_predictions: true
  compliance_frameworks: [fca_consumer_duty, eu_ai_act]

  # Workstream #3 — tamper evidence.
  tamper_evidence: true
  signing_key_env: SENTINEL_AUDIT_KEY    # or…
  signing_key_path: /etc/sentinel/audit.key
```

| Field | Type | Default | Description |
|---|---|---|---|
| `tamper_evidence` | bool | `false` | When `true`, attaches HMAC + chain pointer to every event. |
| `signing_key_env` | string | `SENTINEL_AUDIT_KEY` | Env var holding the signing key bytes. |
| `signing_key_path` | string \| null | `null` | File holding the signing key. Mutually exclusive with `signing_key_env`. |

The keystore must resolve to non-empty bytes at `SentinelClient`
construction time, or `ConfigValidationError` is raised. File-backed
keystores warn (and refuse to load in strict deployments) if the file
mode is not `0600`.

> The keystore is a pluggable `BaseKeystore` ABC. Workstream #2 will
> add an `AzureKeyVaultKeystore` that drops in here without any other
> changes.

### Generating a key

```bash
# 32 bytes of cryptographically random hex
SENTINEL_AUDIT_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
export SENTINEL_AUDIT_KEY
```

For long-lived deployments, store the key in the secret manager of
your choice and inject it via the env var or a 0600-mode file at boot.
Treat key rotation the same as any other HMAC key rotation: write a
new event under the old key first, snapshot the chain head, then
re-key.

### Verifying integrity

```bash
sentinel audit verify --config sentinel.yaml
# OK — 1,402 events / 1,402 signed / chain head: 9b2f4c…

sentinel audit verify --config sentinel.yaml --since 2026-04-01
# Restrict the verification window to events on or after 2026-04-01.
```

`sentinel audit verify` walks every JSON-lines file under
`audit.path` in chronological order, recomputes each `event_hmac`,
and validates the `previous_hash` chain. Files that predate the
keystore are reported as `unsigned`, not `corrupted`. Detected issues
are categorised as:

- `hash_mismatch` — the event's HMAC does not match its content.
- `broken_chain` — `previous_hash` does not match the previous event.
- `missing_hmac` — the event has no HMAC but tamper evidence is on.
- `unknown_key` — the HMAC doesn't validate under the configured key.
- `unparsable` — the line could not be parsed as JSON.

The command exits `0` when the chain is intact and `1` when any
tampering is detected. Wire it into your nightly cron and alert on
non-zero exits.

### Inspecting the chain head

```bash
sentinel audit chain-info --config sentinel.yaml
# {
#   "tamper_evidence": true,
#   "audit_path": "/var/lib/sentinel/audit",
#   "chain_head": "9b2f4c8a…",
#   "key_fingerprint": "f0aa2c91"
# }
```

The `key_fingerprint` is the first 8 hex chars of `SHA-256(signing
key)` — safe to log so you can correlate which key signed which
trail without ever exposing the key itself.

---

## 2. Dashboard authentication

The dashboard supports three auth modes via `dashboard.server.auth`:

| Mode | Use case |
|---|---|
| `none` | Local-first dev, single-user laptop. Default. |
| `basic` | Single shared credential behind a VPN; also accepts username → role mapping for RBAC. |
| `bearer` | API tokens, CI bots, monitoring scrapers; validates JWTs against a JWKS URL. |

### HTTP Basic

```yaml
dashboard:
  server:
    auth: basic
    basic_auth_username: ml-ops
    basic_auth_password: ${SENTINEL_DASHBOARD_PASSWORD}
```

The password is stored as `pydantic.SecretStr` and never appears in
`config show`, `model_dump()`, or structlog output. The dependency
uses `secrets.compare_digest` for timing-safe comparison and returns
`401` with `WWW-Authenticate: Basic realm="Sentinel"` on failure.

### Bearer JWT

```yaml
dashboard:
  server:
    auth: bearer
    bearer:
      jwks_url: https://login.example.com/.well-known/jwks.json
      issuer: https://login.example.com/
      audience: sentinel-dashboard
      username_claim: sub        # default
      roles_claim: roles         # default
      algorithms: [RS256]        # default
      cache_ttl_seconds: 3600    # JWKS cache lifetime
      leeway_seconds: 30         # clock skew allowance for exp/nbf
```

> Bearer mode requires the `[dashboard]` extra (`pip install
> "sentinel-mlops[dashboard]"`) which pulls in `PyJWT` and `httpx`.

The JWKS document is cached in-memory by `kid` and refreshed on cache
miss. Tokens are validated for signature, `exp`, `nbf`, `iss`, and
`aud`. Failures return `401` with
`WWW-Authenticate: Bearer error="invalid_token"`.

> **Out of scope for workstream #3**: Authorization Code + PKCE,
> session cookies, OIDC discovery, logout, and trusted reverse-proxy
> header trust. The Bearer flow is intentionally a thin "validate the
> JWT my IdP minted for me" layer, not a full OIDC client.

### Auth dispatch and the `Principal`

All three modes funnel through `build_auth_middleware()` in
`sentinel/dashboard/security/auth.py`, which sets
`request.state.principal` to a frozen `Principal`:

```python
@dataclass(frozen=True)
class Principal:
    username: str
    roles: frozenset[str]
    permissions: frozenset[str]
    auth_mode: str
```

The principal is the only thing every downstream piece of the dashboard
sees — RBAC, audit logging, the principal-aware rate limit bucket, etc.
`auth: none` injects an `ANONYMOUS_PRINCIPAL` so unauthenticated
local-first runs still have a coherent principal.

---

## 3. RBAC

### Roles, permissions, and routes

When `dashboard.server.rbac.enabled: true`, every route in
`pages.py` and `api.py` is gated by a permission. Permissions are
namespaced (`drift.read`, `audit.verify`, `deployments.promote`, …)
and roles aggregate them. Roles inherit transitively along
`role_hierarchy`, so an `operator` automatically holds every
`viewer` permission.

### Default role matrix

```yaml
dashboard:
  server:
    rbac:
      enabled: true
      default_role: viewer
      role_hierarchy: [viewer, operator, admin]
      role_permissions:
        viewer:
          - drift.read
          - features.read
          - registry.read
          - audit.read
          - llmops.read
          - agentops.read
          - deployments.read
          - compliance.read
        operator:
          - audit.verify
          - deployments.promote
          - retrain.trigger
          - golden.run
        admin:
          - "*"          # wildcard — all permissions
      users:
        - username: alice
          roles: [admin]
        - username: bob
          roles: [operator]
        - username: carol
          roles: [viewer]
```

### Per-route guards

The route → permission table lives in
`sentinel/dashboard/security/route_perms.py`. Adding a new dashboard
route is a one-line change there; the guard wires up automatically
via `Depends(require_permission("..."))`.

### RBAC disabled

When `rbac.enabled: false` (the default), `require_permission` is a
no-op and every authenticated principal sees every route. This keeps
the local-first dev experience intact while letting enterprise
customers opt in.

---

## 4. CSRF protection

The dashboard is read-only today. Workstream #6 will add
write actions (approve a retrain, promote a deployment, run a golden
suite). CSRF protection is installed now so #6 can ship safely
without re-architecting the auth surface.

### Pattern

Double-submit cookie. On every response, the middleware sets a
`sentinel_csrf` cookie containing a per-request token. On unsafe
methods (POST/PUT/PATCH/DELETE) the middleware verifies that the
cookie matches the `X-CSRF-Token` header (or `csrf_token` form
field). On mismatch: `403 CSRFError`.

### Exemptions

- `GET`, `HEAD`, `OPTIONS`, `TRACE` — never gated.
- `Authorization: Bearer …` requests — the bearer token *is* the
  auth, there is no cookie to forge.
- `/static/*` — always exempt.

### HTMX integration

`templates/base.html` injects:

```html
<meta name="csrf-token" content="{{ csrf_token }}">
<script>
  document.body.addEventListener("htmx:configRequest", (e) => {
    e.detail.headers["X-CSRF-Token"] =
      document.querySelector('meta[name="csrf-token"]').content;
  });
</script>
```

Every HTMX request — present and future — automatically includes the
token. Vanilla `fetch()` calls have to add the header themselves.

### Configuration

```yaml
dashboard:
  server:
    csrf:
      enabled: true                # default true
      cookie_name: sentinel_csrf
      header_name: X-CSRF-Token
      cookie_secure: null          # null = auto-detect from request scheme
      cookie_samesite: lax         # lax | strict | none
```

---

## 5. Rate limiting

In-memory token-bucket rate limiter. Per-IP for unauthenticated
requests, per-username for authenticated ones. Three bucket groups:

| Group | Routes | Default | Why |
|---|---|---|---|
| `default` | Dashboard pages | 100/min | Page nav + HTMX swaps |
| `api` | `/api/*` | 300/min | Polling endpoints, dashboard refresh |
| `auth` | Login surface | 10/min | Anti-brute-force on basic auth |

Excludes `/static/*` and the health endpoint. Returns `429
Too Many Requests` with `Retry-After` headers when a bucket is
empty.

```yaml
dashboard:
  server:
    rate_limit:
      enabled: true
      default_per_minute: 100
      api_per_minute: 300
      auth_per_minute: 10
      burst_multiplier: 2.0   # bucket capacity = limit × multiplier
```

> **Single-worker assumption.** Token buckets live in process memory
> and don't survive restart. This is intentional — Sentinel targets
> single-worker production deployments and the local-first dev
> experience. If you need a distributed limiter, put a reverse proxy
> with rate limiting in front.

---

## 6. Security response headers

`SecurityHeadersMiddleware` adds the following to every response:

| Header | Value | Notes |
|---|---|---|
| `X-Frame-Options` | `DENY` | Blocks clickjacking. |
| `X-Content-Type-Options` | `nosniff` | Stops MIME-type sniffing. |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Conservative default. |
| `Permissions-Policy` | `geolocation=(), microphone=()` | No surface area for these features. |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | Only when the request was HTTPS. |
| `Content-Security-Policy` | dashboard-friendly default | Allows the Tailwind / HTMX / Plotly CDNs the dashboard already loads. |

CSP is overridable:

```yaml
dashboard:
  server:
    csp:
      enabled: true
      policy: "default-src 'self'; script-src 'self' https://cdn.example.com"
```

Set `csp.enabled: false` to drop the CSP header entirely (e.g. when
your reverse proxy already injects one).

---

## 7. Middleware order

When you start the dashboard with `sentinel dashboard --config
sentinel.yaml`, `create_dashboard_app` registers the full middleware
stack in this order (outermost → innermost):

1. **`SecurityHeadersMiddleware`** — always on, costs nothing, sets
   the response headers above.
2. **`RateLimitMiddleware`** — block flood traffic before doing real
   work.
3. **`CSRFMiddleware`** — block forged writes before the auth lookup.
4. **Auth dispatch** (`basic` / `bearer` / `none`) — populates
   `request.state.principal`.
5. **RBAC checks** — happen in per-route
   `Depends(require_permission(...))`.

Each middleware is independently configurable; setting any of
`csrf.enabled: false`, `rate_limit.enabled: false`, or
`csp.enabled: false` makes that middleware a pass-through for that
deployment.

---

## 8. Signed configs

### Why

A gitops pipeline can sign-and-merge a `sentinel.yaml` at PR time
(reviewed by ML, compliance, and risk). Without verification, a
compromised host could silently flip a threshold or swap a webhook
URL between PR merge and process boot. Signed configs close that gap.

### How

`sentinel/config/signing.py` computes an HMAC-SHA256 over the
**resolved** config (after `extends:` resolution and `${VAR}`
substitution) so signed configs survive inheritance chains and env-var
indirection. The signature is written to a sidecar `<file>.sig` file
containing:

```json
{
  "version": 1,
  "algorithm": "hmac-sha256",
  "signature": "f23a…",
  "digest": "9b2f…",
  "signed_at": "2026-04-01T12:34:56+00:00",
  "key_fingerprint": "f0aa2c91"
}
```

The canonicalisation step strips loader bookkeeping (`__source__`,
`__sources__`) and emits sorted-key JSON, so two semantically
identical configs always produce the same signature regardless of
YAML formatting.

### Signing a config

```bash
export SENTINEL_CONFIG_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
sentinel config sign --config sentinel.yaml
# signed sentinel.yaml → sentinel.yaml.sig
#   key fingerprint: f0aa2c91
#   digest:          9b2f4c8a…
#   signed at:       2026-04-01T12:34:56+00:00
```

Use `--out` to write the sidecar elsewhere, `--key-file` to read the
key from a 0600-mode file instead of an env var:

```bash
sentinel config sign \
  --config sentinel.yaml \
  --key-file /etc/sentinel/config.key \
  --out /etc/sentinel/sentinel.yaml.sig
```

### Verifying a config

```bash
sentinel config verify-signature --config sentinel.yaml
# OK — sentinel.yaml matches signature
#   key fingerprint: f0aa2c91
#   signed at:       2026-04-01T12:34:56+00:00
```

Exits `0` on a valid signature and `1` on tampering, missing sidecar,
or wrong key. Use it as a CI check on the config repo and as a
pre-deploy gate on the host.

### Programmatic loader integration

```python
from sentinel.config.loader import ConfigLoader
from sentinel.foundation.audit.keystore import EnvKeystore

loader = ConfigLoader(
    "sentinel.yaml",
    verify_signature=True,
    signature_keystore=EnvKeystore("SENTINEL_CONFIG_KEY"),
)
cfg = loader.load()
assert loader.signature_verified
```

`verify_signature=True` without a `signature_keystore` raises
`ConfigSignatureError` at construction time. A missing or
mismatched signature raises the same error from `loader.load()`.

### Enforcing signing on the dashboard

```yaml
dashboard:
  server:
    require_signed_config: true
```

When this flag is set, `sentinel dashboard --config sentinel.yaml`
re-loads the config with `verify_signature=True` using a signing
keystore picked from the `--config-key-env` (default
`SENTINEL_CONFIG_KEY`) or `--config-key-file` CLI option, and
refuses to start if the signature does not match.

```bash
SENTINEL_CONFIG_KEY=$(cat /etc/sentinel/config.key) \
  sentinel dashboard --config sentinel.yaml
```

---

## 9. Putting it all together

A production-shaped dashboard config that opts into every workstream
#3 hardening looks like this:

```yaml
audit:
  storage: local
  path: /var/lib/sentinel/audit/
  retention_days: 2555
  tamper_evidence: true
  signing_key_path: /etc/sentinel/audit.key

dashboard:
  enabled: true
  server:
    host: 127.0.0.1
    port: 8000
    auth: bearer
    bearer:
      jwks_url: https://login.example.com/.well-known/jwks.json
      issuer: https://login.example.com/
      audience: sentinel-dashboard
    rbac:
      enabled: true
      default_role: viewer
      users:
        - username: alice
          roles: [admin]
        - username: bob
          roles: [operator]
    csrf:
      enabled: true
    rate_limit:
      enabled: true
      default_per_minute: 100
      api_per_minute: 300
    csp:
      enabled: true
    require_signed_config: true
```

Boot with both keys exported:

```bash
export SENTINEL_AUDIT_KEY=$(cat /etc/sentinel/audit.key)
export SENTINEL_CONFIG_KEY=$(cat /etc/sentinel/config.key)
sentinel config sign --config sentinel.yaml          # before each deploy
sentinel audit verify --config sentinel.yaml         # nightly cron
sentinel dashboard --config sentinel.yaml            # production process
```

---

## 10. Out of scope (deferred to later workstreams)

- **Azure Key Vault keystore** for both audit signing and config
  signing → workstream #2 (Azure integrations). The `BaseKeystore`
  ABC built here is the integration point.
- **Authorization Code + PKCE OIDC flow**, session cookies, logout —
  Bearer-only is the only OIDC surface in #3.
- **Trusted reverse-proxy header trust** (`X-Forwarded-User`).
- **Audit-trail backends other than local filesystem** (Azure Blob,
  S3, GCS) → workstream #2.
- **Distributed rate limiting** (Redis, dragonfly) → out of scope,
  single-worker assumption holds.
- **Dashboard write actions / approval workflows** → workstream #6.
  This plan installs the CSRF + RBAC plumbing #6 will build on.
- **Self-instrumentation of the security middleware** with Prometheus
  / OTel metrics → workstream #4.
