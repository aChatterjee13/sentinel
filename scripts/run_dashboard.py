"""Launch the Sentinel dashboard with a fully seeded demo client.

This script is the fastest way to *see* the dashboard:

    pip install -e ".[all,dashboard]"
    python scripts/run_dashboard.py

The ``all`` extra is intentionally light — it does **not** pull
``sentence-transformers`` / ``torch`` / ``presidio``, so the install
resolves on Python 3.13 (where torch has no wheels) and inside
lightweight CI images. If you also want embedding-based semantic drift,
jailbreak detection, topic fencing, or PII analysis, install the heavier
``ml-extras`` extra on top:

    pip install -e ".[all,dashboard,ml-extras]"

It builds an in-memory ``SentinelClient`` with every subsystem enabled
(LLMOps + AgentOps + Dashboard), seeds enough realistic data so that every
page in the navigation has something to render, and then boots the FastAPI
app on ``http://127.0.0.1:8000``.

Nothing is permanent — the registry, audit trail, and prompt files all live
under a temporary directory printed at the top of the run. Delete the
directory to start fresh.

Pages to visit once the server is up:

    /                       Overview (status, recent alerts, deployments)
    /drift                  Drift timeline + reports
    /features               Feature health table
    /registry               Model versions
    /registry/{m}/{v}       Per-version detail
    /audit                  Filterable audit trail
    /llmops/prompts         Prompt versions
    /llmops/guardrails      Guardrail violations
    /llmops/tokens          Token economics + cost trend
    /agentops/traces        Recent agent traces
    /agentops/traces/{id}   Trace timeline + tree view
    /agentops/tools         Tool call stats
    /agentops/agents        Agent registry
    /deployments            Active + historical deployments
    /compliance             Compliance framework coverage

Press Ctrl+C to stop the server.
"""

from __future__ import annotations

import argparse
import contextlib
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Make ``sentinel`` importable when running from the repo root without an
# editable install (e.g. `python scripts/run_dashboard.py`).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentinel.agentops.agent_registry import AgentSpec  # noqa: E402
from sentinel.config.schema import (  # noqa: E402
    AgentOpsConfig,
    AlertsConfig,
    AuditConfig,
    ChannelConfig,
    DashboardConfig,
    DashboardServerConfig,
    DashboardUIConfig,
    DataDriftConfig,
    DriftConfig,
    LLMOpsConfig,
    ModelConfig,
    SentinelConfig,
    TokenEconomicsConfig,
)
from sentinel.core.client import SentinelClient  # noqa: E402
from sentinel.foundation.registry.backends.local import LocalRegistryBackend  # noqa: E402
from sentinel.foundation.registry.model_registry import ModelRegistry  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Config + client construction
# ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "claims_fraud_v2"
DOMAIN = "tabular"


def build_config(workspace: Path) -> SentinelConfig:
    """Build a fully-enabled SentinelConfig rooted under ``workspace``."""
    audit_dir = workspace / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    return SentinelConfig(
        model=ModelConfig(
            name=MODEL_NAME,
            domain=DOMAIN,
            type="classification",
            framework="xgboost",
            version="1.0.0",
        ),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
        ),
        alerts=AlertsConfig(
            channels=[
                ChannelConfig(
                    type="slack",
                    webhook_url="https://example.invalid/dashboard-demo",
                    channel="#ml-alerts",
                ),
            ],
        ),
        audit=AuditConfig(
            storage="local",
            path=str(audit_dir),
            retention_days=2555,
            log_predictions=False,
            log_explanations=False,
            compliance_frameworks=["fca_consumer_duty", "eu_ai_act", "pra_ss123"],
        ),
        llmops=LLMOpsConfig(
            enabled=True,
            mode="rag",
            token_economics=TokenEconomicsConfig(
                track_by=["model", "prompt_version"],
                pricing={
                    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                    "gpt-4o": {"input": 5.00, "output": 15.00},
                },
            ),
        ),
        agentops=AgentOpsConfig(enabled=True),
        dashboard=DashboardConfig(
            enabled=True,
            server=DashboardServerConfig(host="127.0.0.1", port=8000),
            ui=DashboardUIConfig(
                title="Sentinel Demo Dashboard",
                theme="auto",
            ),
        ),
    )


def build_client(workspace: Path) -> SentinelClient:
    """Build a SentinelClient backed by an isolated workspace."""
    cfg = build_config(workspace)
    registry_root = workspace / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(backend=LocalRegistryBackend(root=registry_root))
    client = SentinelClient(cfg, registry=registry)
    _install_fake_embedder(client)
    return client


def _install_fake_embedder(client: SentinelClient) -> None:
    """Inject a deterministic fake embed function into the semantic drift monitor.

    The real monitor lazy-imports ``sentence_transformers`` on first use. If
    the package is installed in the host environment (as in many CI and dev
    setups), that import can take 20-40 seconds and may trigger a model
    download from HuggingFace on first call. For a seed-and-launch script
    that runs in under a second, we don't need real embeddings — a hash-based
    fake is plenty to exercise the code paths without touching disk or net.
    """
    if client.llmops is None:
        return

    def _fake_embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            # Deterministic per-text seed — gives stable pseudo-embeddings.
            seed = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(seed)
            out.append(rng.standard_normal(32).tolist())
        return out

    client.llmops.semantic_drift._embed_fn = _fake_embed


# ─────────────────────────────────────────────────────────────────────
# Seed data
# ─────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "amount_log",
    "merchant_cc",
    "txn_velocity",
    "geo_distance_km",
    "device_age_days",
    "account_age_months",
]


def _make_dataframe(n: int, drifted: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = []
    for i in range(len(FEATURE_NAMES)):
        if drifted and i in (0, 2):
            cols.append(rng.normal(loc=1.6, scale=1.4, size=n))
        else:
            cols.append(rng.normal(loc=0.0, scale=1.0, size=n))
    return np.stack(cols, axis=1)


def seed_registry(client: SentinelClient) -> None:
    """Register a few model versions with metadata + baselines."""
    versions = [
        (
            "1.0.0",
            {
                "framework": "xgboost",
                "trained_on": "2025-09-15",
                "metrics": {"accuracy": 0.91, "f1": 0.88, "auc": 0.94},
                "tags": ["production", "baseline"],
            },
        ),
        (
            "1.1.0",
            {
                "framework": "xgboost",
                "trained_on": "2025-12-01",
                "metrics": {"accuracy": 0.93, "f1": 0.90, "auc": 0.95},
                "tags": ["production"],
            },
        ),
        (
            "1.2.0",
            {
                "framework": "xgboost",
                "trained_on": "2026-03-20",
                "metrics": {"accuracy": 0.94, "f1": 0.91, "auc": 0.96},
                "tags": ["canary"],
            },
        ),
    ]
    for version, metadata in versions:
        client.registry.register(MODEL_NAME, version, **metadata)
    # Mark v1.1.0 as the production baseline so registry detail has data.
    with contextlib.suppress(Exception):
        client.registry.promote(MODEL_NAME, "1.1.0", status="production")
    client.model_version = "1.2.0"


def seed_drift_history(client: SentinelClient) -> None:
    """Fit a baseline, log predictions, then run drift checks (clean + drifted)."""
    baseline = _make_dataframe(800, drifted=False, seed=11)
    client.fit_baseline(baseline)

    # Clean window — drift check should pass.
    clean = _make_dataframe(400, drifted=False, seed=22)
    for row in clean[:200]:
        features = {name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)}
        client.log_prediction(features=features, prediction=0)
    with contextlib.suppress(Exception):
        client.check_drift()
    client.clear_buffer()

    # Drifted window — should fire alerts.
    drifted = _make_dataframe(400, drifted=True, seed=33)
    for row in drifted[:200]:
        features = {name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)}
        client.log_prediction(features=features, prediction=1)
    with contextlib.suppress(Exception):
        client.check_drift()

    # Build a feature-health snapshot too so /features renders.
    with contextlib.suppress(Exception):
        client.get_feature_health()


def seed_audit_extras(client: SentinelClient) -> None:
    """Drop a handful of mixed audit events so /audit has variety to filter."""
    events = [
        ("model_registered", {"actor": "ml-team@bank.com"}),
        ("alert_sent", {"severity": "high", "channel": "slack", "subject": "psi=0.27 > 0.20"}),
        ("alert_sent", {"severity": "info", "channel": "slack", "subject": "canary 25→50"}),
        ("retrain_triggered", {"reason": "drift_confirmed", "pipeline": "azureml://retrain"}),
        (
            "deployment_started",
            {
                "strategy": "canary",
                "from_version": "1.1.0",
                "to_version": "1.2.0",
                "traffic_pct": 5,
            },
        ),
        ("compliance_review", {"framework": "fca_consumer_duty", "outcome": "pass"}),
        ("compliance_review", {"framework": "eu_ai_act", "outcome": "pass"}),
    ]
    for event_type, payload in events:
        client.audit.log(
            event_type=event_type,
            model_name=MODEL_NAME,
            model_version="1.2.0",
            **payload,
        )


def seed_llmops(client: SentinelClient) -> None:
    """Register prompts, log LLM calls, fire one guardrail violation."""
    if client.llmops is None:
        return
    pm = client.llmops.prompts
    pm.register(
        name="claims_qa",
        version="1.0",
        system_prompt="You are an insurance claims analyst. Answer briefly.",
        template="Claim: {{claim_text}}\n\nQuestion: {{question}}",
        few_shot_examples=[
            {
                "user": "Is water damage from a burst pipe covered?",
                "assistant": "Yes — covered under the household sudden-escape clause.",
            },
        ],
        metadata={"author": "ml-team", "reviewed_by": "compliance"},
        traffic_pct=90,
    )
    pm.register(
        name="claims_qa",
        version="1.1",
        system_prompt="You are an insurance claims analyst. Answer concisely with citations.",
        template="Claim: {{claim_text}}\n\nQuestion: {{question}}\n\nCite source chunks.",
        metadata={"author": "ml-team", "reviewed_by": "compliance"},
        traffic_pct=10,
    )
    pm.register(
        name="fraud_triage",
        version="1.0",
        system_prompt="You are a fraud triage assistant.",
        template="Score this transaction for fraud risk:\n{{transaction}}",
        metadata={"author": "ml-team"},
        traffic_pct=100,
    )

    # Token economics — log a spread of LLM calls so the trend chart has data.
    rng = random.Random(7)
    models = ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini", "gpt-4o"]
    for _ in range(40):
        model = rng.choice(models)
        in_tok = rng.randint(180, 900)
        out_tok = rng.randint(80, 400)
        latency = rng.uniform(280, 1800)
        client.llmops.token_tracker.record(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
            cached=rng.random() < 0.15,
            prompt_version="1.0" if rng.random() < 0.7 else "1.1",
            user_segment=rng.choice(["gold", "silver", "bronze"]),
        )

    # A few full LLM call logs (which also exercise log_call → audit).
    for i in range(6):
        client.llmops.log_call(
            prompt_name="claims_qa",
            prompt_version="1.0",
            query="Is water damage from a burst pipe covered?",
            response="Yes — the household policy covers sudden escape of water.",
            context_chunks=["Section 4.2 — sudden escape of water is covered."],
            model="gpt-4o-mini",
            input_tokens=420,
            output_tokens=130,
            latency_ms=540 + i * 22,
            guardrail_results={},
            user_id=f"user-{i:03d}",
        )

    # Two synthetic guardrail violations so /llmops/guardrails renders.
    client.audit.log(
        event_type="llmops.guardrail_violation",
        model_name=MODEL_NAME,
        guardrail="pii_detection",
        action="redact",
        entities=["account_number"],
        prompt="claims_qa@1.0",
    )
    client.audit.log(
        event_type="llmops.guardrail_violation",
        model_name=MODEL_NAME,
        guardrail="jailbreak_detection",
        action="block",
        score=0.92,
        prompt="claims_qa@1.0",
    )


def seed_agentops(client: SentinelClient) -> None:
    """Register agents, run a couple of traces, record tool calls."""
    if client.agentops is None:
        return

    # Agent registry — two agents, one with two versions.
    client.agentops.registry.register(
        AgentSpec(
            name="claims_processor",
            version="1.0.0",
            description="Processes incoming claims end-to-end.",
            capabilities=["claim_intake", "policy_lookup", "summarisation"],
            tools=["sharepoint_search", "policy_lookup", "llm_extraction"],
            llm_config={"model": "gpt-4o-mini", "temperature": 0.0},
            budget={"max_tokens_per_run": 30000, "max_cost_per_run": 2.0},
            safety_policies={"max_iterations": 50},
            dependencies=[],
            metadata={"team": "claims-platform"},
            baselines={"task_completion_rate": 0.92, "avg_cost_usd": 0.18},
            health_status="healthy",
        )
    )
    client.agentops.registry.register(
        AgentSpec(
            name="claims_processor",
            version="1.1.0",
            description="Processes incoming claims end-to-end.",
            capabilities=["claim_intake", "policy_lookup", "summarisation", "fraud_check"],
            tools=["sharepoint_search", "policy_lookup", "llm_extraction", "fraud_score"],
            llm_config={"model": "gpt-4o-mini", "temperature": 0.0},
            budget={"max_tokens_per_run": 35000, "max_cost_per_run": 2.5},
            safety_policies={"max_iterations": 60},
            dependencies=[],
            metadata={"team": "claims-platform"},
            baselines={"task_completion_rate": 0.94, "avg_cost_usd": 0.21},
            health_status="healthy",
        )
    )
    client.agentops.registry.register(
        AgentSpec(
            name="underwriting_assistant",
            version="0.9.0",
            description="Helps underwriters score new policies.",
            capabilities=["risk_scoring", "actuarial_lookup"],
            tools=["risk_database", "actuarial_tables"],
            llm_config={"model": "gpt-4o", "temperature": 0.1},
            budget={"max_tokens_per_run": 50000, "max_cost_per_run": 5.0},
            safety_policies={"max_iterations": 40},
            dependencies=[],
            metadata={"team": "underwriting"},
            baselines={"task_completion_rate": 0.88, "avg_cost_usd": 0.84},
            health_status="degraded",
        )
    )

    # Run a few traces so /agentops/traces and /agentops/traces/{id} populate.
    tracer = client.agentops.tracer
    for run_id in range(3):
        with tracer.trace("claims_processor", claim_id=f"CLM-{1000 + run_id}") as _ctx:
            with tracer.span("plan", kind="reasoning"):
                time.sleep(0.005)
            with tracer.span("tool_call", kind="tool", tool="sharepoint_search"):
                time.sleep(0.007)
            with tracer.span("tool_call", kind="tool", tool="policy_lookup"):
                time.sleep(0.004)
            with tracer.span("synthesise", kind="reasoning"):
                time.sleep(0.006)
            with tracer.span("respond", kind="output"):
                time.sleep(0.002)

    # Record tool stats directly so /agentops/tools has data even if the
    # tracer didn't already pipe everything through the monitor.
    monitor = client.agentops.tool_monitor
    rng = random.Random(13)
    for _ in range(15):
        monitor.record(
            agent="claims_processor",
            tool="sharepoint_search",
            inputs={"query": "policy 12345"},
            output={"chunks": 4},
            success=True,
            latency_ms=rng.uniform(180, 460),
        )
    for _ in range(8):
        monitor.record(
            agent="claims_processor",
            tool="policy_lookup",
            inputs={"policy_id": "P-9001"},
            output={"coverage": "household"},
            success=True,
            latency_ms=rng.uniform(80, 240),
        )
    monitor.record(
        agent="claims_processor",
        tool="policy_lookup",
        inputs={"policy_id": "P-MISSING"},
        output=None,
        success=False,
        error="not found",
        latency_ms=120.0,
    )
    for _ in range(4):
        monitor.record(
            agent="underwriting_assistant",
            tool="risk_database",
            inputs={"applicant_id": "A-44"},
            output={"score": 0.31},
            success=True,
            latency_ms=rng.uniform(220, 540),
        )


def seed_deployments(client: SentinelClient) -> None:
    """Kick off a canary deployment so /deployments has an active row."""
    with contextlib.suppress(Exception):
        client.deployment_manager.start(
            model_name=MODEL_NAME,
            to_version="1.2.0",
            from_version="1.1.0",
            strategy_override="canary",
        )


def seed_all(client: SentinelClient) -> None:
    """Run every seeder in dependency order."""
    seed_registry(client)
    seed_drift_history(client)
    seed_audit_extras(client)
    seed_llmops(client)
    seed_agentops(client)
    seed_deployments(client)


# ─────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Sentinel dashboard with seeded demo data.")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    p.add_argument(
        "--workspace",
        default=None,
        help="directory for audit + registry files (default: a fresh tempdir)",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="enable uvicorn reload (developer-only, slow first request)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    workspace = (
        Path(args.workspace) if args.workspace else Path(tempfile.mkdtemp(prefix="sentinel-dash-"))
    )
    workspace.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Sentinel demo dashboard")
    print("=" * 70)
    print(f" workspace : {workspace}")
    print(f" model     : {MODEL_NAME}")
    print(f" url       : http://{args.host}:{args.port}")
    print("-" * 70)
    print(" Building client and seeding demo data …")

    client = build_client(workspace)
    seed_all(client)

    print(" Seed complete. Pages to try:")
    for path in [
        "/",
        "/drift",
        "/features",
        "/registry",
        "/audit",
        "/llmops/prompts",
        "/llmops/guardrails",
        "/llmops/tokens",
        "/agentops/traces",
        "/agentops/tools",
        "/agentops/agents",
        "/deployments",
        "/compliance",
    ]:
        print(f"   http://{args.host}:{args.port}{path}")
    print("-" * 70)
    print(" Starting uvicorn — press Ctrl+C to stop.")
    print("=" * 70)

    # Import inside main so users without the [dashboard] extra get a clean
    # error message via the same exception path the CLI uses.
    try:
        from sentinel.dashboard.server import run as run_dashboard
    except ImportError as exc:
        raise SystemExit(
            "Dashboard requires the [dashboard] extra. Install with:\n"
            '  pip install -e ".[all,dashboard]"\n'
            f"(import error: {exc})"
        ) from exc

    run_dashboard(client, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
