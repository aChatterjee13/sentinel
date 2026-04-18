"""Sentinel CLI — `sentinel init`, `check`, `status`, `deploy`, `audit`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml

from sentinel import __version__
from sentinel.config.defaults import default_yaml_template
from sentinel.config.loader import ConfigLoader, load_config
from sentinel.config.references import validate_file_references
from sentinel.config.secrets import masked_dump
from sentinel.config.signing import (
    read_signature_file,
    sign_config,
    verify_config,
    write_signature_file,
)
from sentinel.core.client import SentinelClient
from sentinel.core.exceptions import (
    AuditKeystoreError,
    ConfigCircularInheritanceError,
    ConfigMissingEnvVarError,
    ConfigSignatureError,
    ConfigValidationError,
)
from sentinel.foundation.audit.keystore import (
    BaseKeystore,
    EnvKeystore,
    FileKeystore,
)


@click.group()
@click.version_option(__version__, prog_name="sentinel")
def cli() -> None:
    """Project Sentinel — Unified MLOps + LLMOps + AgentOps SDK."""


# ── init ───────────────────────────────────────────────────────────


@cli.command()
@click.option("--name", default="my_model", help="Model name to seed the config with")
@click.option("--out", default="sentinel.yaml", help="Output path")
@click.option("--force", is_flag=True, help="Overwrite if the file exists")
@click.option(
    "--ci",
    type=click.Choice(["azure-devops"]),
    default=None,
    help="Generate a CI pipeline scaffold alongside the config",
)
def init(name: str, out: str, force: bool, ci: str | None) -> None:
    """Generate a starter sentinel.yaml config."""
    out_path = Path(out)
    if out_path.exists() and not force:
        click.echo(f"refusing to overwrite {out_path} (use --force)", err=True)
        sys.exit(1)
    out_path.write_text(default_yaml_template(name))
    click.echo(f"wrote {out_path}")

    if ci == "azure-devops":
        _generate_azure_devops_pipeline(name, out_path.parent)


# ── check ──────────────────────────────────────────────────────────


@cli.command()
@click.option("--config", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option(
    "--reference", type=click.Path(exists=True), help="Reference dataset (CSV/Parquet/JSONL)"
)
@click.option("--current", type=click.Path(exists=True), help="Current dataset to check")
def check(config: str, reference: str | None, current: str | None) -> None:
    """Run a one-off drift check using the reference & current datasets."""
    client = SentinelClient.from_config(config)
    if reference:
        client.fit_baseline(_load_dataset(reference))
    report = client.check_drift(_load_dataset(current)) if current else client.check_drift()
    click.echo(json.dumps(report.model_dump(mode="json"), indent=2, default=str))


# ── status ─────────────────────────────────────────────────────────


@cli.command()
@click.option("--config", default="sentinel.yaml", help="Path to sentinel.yaml")
def status(config: str) -> None:
    """Show current Sentinel status for the configured model."""
    client = SentinelClient.from_config(config)
    click.echo(json.dumps(client.status(), indent=2, default=str))


# ── deploy ─────────────────────────────────────────────────────────


@cli.command()
@click.option("--config", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option("--version", required=True, help="Model version to deploy")
@click.option("--strategy", default=None, help="Deployment strategy override")
@click.option("--traffic", default=None, type=int, help="Initial traffic %")
@click.option("--dry-run", is_flag=True, default=False, help="Validate without deploying")
def deploy(
    config: str,
    version: str,
    strategy: str | None,
    traffic: int | None,
    dry_run: bool,
) -> None:
    """Begin a deployment for the given model version."""
    client = SentinelClient.from_config(config)
    if dry_run:
        click.echo(json.dumps({
            "dry_run": True,
            "version": version,
            "strategy": strategy or client.config.deployment.strategy,
            "traffic_pct": traffic,
            "target": client.config.deployment.target,
            "validation": "passed",
        }, indent=2))
        return
    state = client.deploy(version=version, strategy=strategy, traffic_pct=traffic)
    click.echo(json.dumps(state.model_dump(mode="json"), indent=2, default=str))


# ── registry ───────────────────────────────────────────────────────


@cli.group()
def registry() -> None:
    """Inspect the model registry."""


@registry.command("list")
@click.option("--config", default="sentinel.yaml")
def registry_list(config: str) -> None:
    """List models and versions."""
    client = SentinelClient.from_config(config)
    models = client.registry.list_models()
    for m in models:
        versions = client.registry.list_versions(m)
        click.echo(f"{m}: {', '.join(versions) if versions else '(no versions)'}")


@registry.command("show")
@click.option("--config", default="sentinel.yaml")
@click.option("--version", required=True)
def registry_show(config: str, version: str) -> None:
    """Show details for a registered model version."""
    client = SentinelClient.from_config(config)
    mv = client.registry.get(client.model_name, version)
    click.echo(json.dumps(mv.model_dump(mode="json"), indent=2, default=str))


# ── audit ──────────────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.option("--config", default="sentinel.yaml")
@click.option("--type", "event_type", default=None, help="Filter by event type")
@click.option("--limit", default=20, type=int)
@click.pass_context
def audit(
    ctx: click.Context,
    config: str,
    event_type: str | None,
    limit: int,
) -> None:
    """Audit trail operations.

    Without a subcommand, ``sentinel audit`` is an alias for
    ``sentinel audit query`` and tails the most recent events. Use
    ``sentinel audit verify`` to check the hash chain and
    ``sentinel audit chain-info`` to inspect the signing key
    fingerprint and chain head.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(audit_query, config=config, event_type=event_type, limit=limit)


@audit.command("query")
@click.option("--config", default="sentinel.yaml")
@click.option("--type", "event_type", default=None, help="Filter by event type")
@click.option("--limit", default=20, type=int)
def audit_query(config: str, event_type: str | None, limit: int) -> None:
    """Tail the audit trail."""
    client = SentinelClient.from_config(config)
    events = list(client.audit.query(event_type=event_type, limit=limit))
    for e in events:
        click.echo(f"{e.timestamp.isoformat()}  {e.event_type:30s}  {e.model_name or '-':25s}")


@audit.command("verify")
@click.option("--config", default="sentinel.yaml")
@click.option("--since", default=None, help="ISO timestamp lower bound")
@click.option("--until", default=None, help="ISO timestamp upper bound")
def audit_verify(config: str, since: str | None, until: str | None) -> None:
    """Verify the audit trail's hash chain.

    Walks every JSON-lines file under the configured audit path and
    recomputes the HMAC of every signed event. Exits 0 when the chain
    is intact and 1 when any tampering or chain break is detected.
    """
    from datetime import datetime

    client = SentinelClient.from_config(config)

    def _parse(value: str | None) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError as e:
            raise click.UsageError(f"could not parse {value!r} as ISO datetime") from e

    report = client.audit.verify_integrity(since=_parse(since), until=_parse(until))
    click.echo(report.summary())
    if not report.ok:
        sys.exit(1)


@audit.command("chain-info")
@click.option("--config", default="sentinel.yaml")
def audit_chain_info(config: str) -> None:
    """Print the audit chain head and signing-key fingerprint.

    The fingerprint is the first 8 hex chars of SHA-256(signing key)
    and never leaks the key itself. The chain head is the HMAC of the
    most recently written signed event.
    """
    client = SentinelClient.from_config(config)
    trail = client.audit
    head = trail.chain_head()
    info: dict[str, object] = {
        "tamper_evidence": client.config.audit.tamper_evidence,
        "audit_path": str(trail.path),
        "chain_head": head,
    }
    if trail._keystore is not None:
        try:
            info["key_fingerprint"] = trail._keystore.fingerprint()
        except Exception as e:  # pragma: no cover - defensive
            info["key_fingerprint"] = f"<error: {e}>"
    else:
        info["key_fingerprint"] = None
    click.echo(json.dumps(info, indent=2))


# ── config group ───────────────────────────────────────────────────


@cli.group()
def config() -> None:
    """Inspect, validate, and dump Sentinel configuration files."""


@config.command("validate")
@click.option("--config", "config_path", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option(
    "--strict",
    is_flag=True,
    help="Treat unset env vars and missing file references as errors",
)
def config_validate(config_path: str, strict: bool) -> None:
    """Validate a sentinel.yaml file without instantiating the client.

    With ``--strict``: every ``${VAR}`` token must resolve and every
    referenced path must exist. Without ``--strict``: missing references
    are printed as warnings but the command exits zero.
    """
    try:
        loader = ConfigLoader(config_path, strict_env=strict)
        cfg = loader.load()
    except (
        ConfigMissingEnvVarError,
        ConfigCircularInheritanceError,
        ConfigValidationError,
    ) as e:
        raise click.ClickException(str(e)) from e

    issues = validate_file_references(cfg, Path(config_path).resolve().parent)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    for issue in issues:
        click.echo(issue.format(), err=True)

    if strict and errors:
        raise click.ClickException(f"file reference validation failed: {len(errors)} error(s)")

    suffix = ""
    if warnings:
        suffix = f" ({len(warnings)} warning(s))"
    click.echo(f"OK — model={cfg.model.name} domain={cfg.model.domain}{suffix}")


@config.command("show")
@click.option("--config", "config_path", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option(
    "--unmask",
    is_flag=True,
    help="Reveal secrets instead of masking them (audit-logged)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format",
)
def config_show(config_path: str, unmask: bool, output_format: str) -> None:
    """Print the resolved (merged + env-substituted) config.

    Secrets are rendered as ``"<REDACTED>"`` by default. Pass
    ``--unmask`` to print the plaintext (audit-logged via the standard
    audit trail when storage is configured).
    """
    try:
        cfg = load_config(config_path)
    except (
        ConfigMissingEnvVarError,
        ConfigCircularInheritanceError,
        ConfigValidationError,
    ) as e:
        raise click.ClickException(str(e)) from e

    if unmask:
        click.echo(
            "warning: --unmask reveals secrets in plaintext; do not paste publicly",
            err=True,
        )

    payload = masked_dump(cfg, unmask=unmask)
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, default=str, sort_keys=False))
    else:
        click.echo(yaml.safe_dump(payload, sort_keys=False))


@config.command("sign")
@click.option("--config", "config_path", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option(
    "--key-env",
    default="SENTINEL_CONFIG_KEY",
    help="Environment variable holding the HMAC signing key",
)
@click.option(
    "--key-file",
    default=None,
    help="Path to a file holding the HMAC signing key (mutually exclusive with --key-env)",
)
@click.option(
    "--out",
    default=None,
    help="Output path for the signature sidecar (defaults to <config>.sig)",
)
def config_sign(
    config_path: str,
    key_env: str,
    key_file: str | None,
    out: str | None,
) -> None:
    """Compute and write a detached HMAC signature for a config file.

    The signature is computed over the **resolved** config (after
    ``extends:`` resolution and ``${VAR}`` substitution) so signed
    configs survive inheritance chains and env-var indirection.
    """
    keystore = _build_signing_keystore(key_env=key_env, key_file=key_file)
    try:
        loader = ConfigLoader(config_path)
        loader.load()  # populates resolved_payload
    except (
        ConfigMissingEnvVarError,
        ConfigCircularInheritanceError,
        ConfigValidationError,
    ) as e:
        raise click.ClickException(str(e)) from e
    payload = loader.resolved_payload
    if payload is None:  # pragma: no cover - load() always populates
        raise click.ClickException("could not resolve config payload before signing")

    try:
        key = keystore.get_key()
    except AuditKeystoreError as e:
        raise click.ClickException(str(e)) from e

    signature = sign_config(payload, key)
    sig_path = write_signature_file(out or config_path, signature)
    click.echo(
        f"signed {config_path} → {sig_path}\n"
        f"  key fingerprint: {signature.key_fingerprint}\n"
        f"  digest:          {signature.digest}\n"
        f"  signed at:       {signature.signed_at}"
    )


@config.command("verify-signature")
@click.option("--config", "config_path", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option(
    "--sig",
    "sig_path",
    default=None,
    help="Path to the signature sidecar (defaults to <config>.sig)",
)
@click.option(
    "--key-env",
    default="SENTINEL_CONFIG_KEY",
    help="Environment variable holding the HMAC signing key",
)
@click.option(
    "--key-file",
    default=None,
    help="Path to a file holding the HMAC signing key",
)
def config_verify_signature(
    config_path: str,
    sig_path: str | None,
    key_env: str,
    key_file: str | None,
) -> None:
    """Verify a config file against its detached signature.

    Exits 0 when the signature is valid and 1 on any tampering, key
    mismatch, or missing signature file.
    """
    keystore = _build_signing_keystore(key_env=key_env, key_file=key_file)
    try:
        loader = ConfigLoader(config_path)
        loader.load()
    except (
        ConfigMissingEnvVarError,
        ConfigCircularInheritanceError,
        ConfigValidationError,
    ) as e:
        raise click.ClickException(str(e)) from e
    payload = loader.resolved_payload
    if payload is None:  # pragma: no cover - load() always populates
        raise click.ClickException("could not resolve config payload before verifying")

    try:
        signature = read_signature_file(sig_path or config_path)
        key = keystore.get_key()
    except (ConfigSignatureError, AuditKeystoreError) as e:
        raise click.ClickException(str(e)) from e

    if not verify_config(payload, signature, key):
        click.echo(
            f"FAIL — signature does not match {config_path}\n"
            f"  signature key fingerprint: {signature.key_fingerprint}\n"
            f"  signed at:                 {signature.signed_at}",
            err=True,
        )
        sys.exit(1)
    click.echo(
        f"OK — {config_path} matches signature\n"
        f"  key fingerprint: {signature.key_fingerprint}\n"
        f"  signed at:       {signature.signed_at}"
    )


def _build_signing_keystore(*, key_env: str, key_file: str | None) -> BaseKeystore:
    """Pick a :class:`BaseKeystore` based on CLI flags."""
    if key_file is not None:
        return FileKeystore(key_file)
    return EnvKeystore(key_env)


@cli.command(name="validate")
@click.option("--config", "config_path", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option(
    "--strict",
    is_flag=True,
    help="Treat unset env vars and missing file references as errors",
)
@click.pass_context
def validate(ctx: click.Context, config_path: str, strict: bool) -> None:
    """Alias for ``sentinel config validate`` (kept for backward compatibility)."""
    ctx.invoke(config_validate, config_path=config_path, strict=strict)


# ── cloud ──────────────────────────────────────────────────────────


@cli.group()
def cloud() -> None:
    """Smoke-test cloud backends referenced by a sentinel.yaml."""


_CLOUD_BACKENDS = ("keyvault", "registry", "audit", "deploy")


@cloud.command("test")
@click.option("--config", "config_path", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option(
    "--only",
    type=click.Choice(_CLOUD_BACKENDS),
    default=None,
    help="Test only a single backend (defaults to all four).",
)
def cloud_test(config_path: str, only: str | None) -> None:
    """Ping every cloud backend configured in a sentinel.yaml and report pass/fail.

    The command exits 0 when every selected backend probe succeeds, 1
    otherwise. It performs lightweight reachability checks only — no
    records are written, no models are deployed, no Key Vault secrets
    are decrypted into plaintext on stdout. Intended for CI and for
    operators diagnosing ``DefaultAzureCredential`` failures before a
    full run.
    """
    import time

    selected = [only] if only else list(_CLOUD_BACKENDS)
    failures: list[str] = []

    def _line(status: str, name: str, detail: str, elapsed_ms: float | None = None) -> None:
        elapsed = f" ({elapsed_ms:.0f}ms)" if elapsed_ms is not None else ""
        click.echo(f"[{status}] {name:<12} {detail}{elapsed}")

    # The Key Vault probe is really a config-load test — ``load_config``
    # runs the ``${azkv:...}`` substitution as a side effect. A
    # successful load means every reference resolved.
    if "keyvault" in selected:
        start = time.monotonic()
        try:
            load_config(config_path)
            elapsed = (time.monotonic() - start) * 1000
            _line("OK", "keyvault", "config loaded, all ${azkv:..} resolved", elapsed)
        except Exception as e:
            _line("FAIL", "keyvault", f"{type(e).__name__}: {e}")
            failures.append("keyvault")
            # If the config itself cannot load, nothing else can be tested.
            raise click.exceptions.Exit(1) from e

    # From here on we need a loaded config. Re-use the same load to
    # avoid re-negotiating Key Vault tokens.
    config = load_config(config_path)

    if "registry" in selected:
        start = time.monotonic()
        try:
            backend = SentinelClient._build_registry_backend(config)
            models = backend.list_models()
            elapsed = (time.monotonic() - start) * 1000
            _line(
                "OK",
                "registry",
                f"{config.registry.backend} backend, {len(models)} model(s)",
                elapsed,
            )
        except Exception as e:
            _line("FAIL", "registry", f"{type(e).__name__}: {e}")
            failures.append("registry")

    if "audit" in selected:
        start = time.monotonic()
        try:
            shipper = SentinelClient._build_audit_shipper(config)
            if hasattr(shipper, "health_check"):
                healthy = bool(shipper.health_check())
                if not healthy:
                    raise RuntimeError("shipper.health_check() returned False")
                detail = f"{config.audit.storage} shipper healthy"
            else:
                detail = f"{config.audit.storage} shipper (no remote probe)"
            elapsed = (time.monotonic() - start) * 1000
            # Drain the background worker thread so short-lived CLI
            # runs do not leak daemon threads into the pytest harness.
            if hasattr(shipper, "close"):
                shipper.close()
            _line("OK", "audit", detail, elapsed)
        except Exception as e:
            _line("FAIL", "audit", f"{type(e).__name__}: {e}")
            failures.append("audit")

    if "deploy" in selected:
        start = time.monotonic()
        try:
            from sentinel.action.deployment.manager import DeploymentManager

            target = DeploymentManager._build_target(config.deployment)
            info = target.describe(config.model.name)
            elapsed = (time.monotonic() - start) * 1000
            detail = f"{config.deployment.target} target {info or 'reachable'}"
            _line("OK", "deploy", detail, elapsed)
        except Exception as e:
            _line("FAIL", "deploy", f"{type(e).__name__}: {e}")
            failures.append("deploy")

    if failures:
        click.echo(
            f"\n{len(failures)} backend(s) failed: {', '.join(failures)}",
            err=True,
        )
        raise click.exceptions.Exit(1)

    click.echo(f"\nAll {len(selected)} backend(s) OK.")


# ── dashboard ──────────────────────────────────────────────────────


@cli.command()
@click.option("--config", default="sentinel.yaml", help="Path to sentinel.yaml")
@click.option("--host", default=None, help="Override dashboard.server.host")
@click.option("--port", default=None, type=int, help="Override dashboard.server.port")
@click.option("--reload", is_flag=True, help="Enable uvicorn auto-reload (dev mode)")
@click.option(
    "--open", "open_browser", is_flag=True, help="Open the dashboard in a browser on boot"
)
@click.option(
    "--config-key-env",
    default="SENTINEL_CONFIG_KEY",
    help="Env var holding the config-signing key (used when require_signed_config is true)",
)
@click.option(
    "--config-key-file",
    default=None,
    help="File holding the config-signing key (used when require_signed_config is true)",
)
def dashboard(
    config: str,
    host: str | None,
    port: int | None,
    reload: bool,
    open_browser: bool,
    config_key_env: str,
    config_key_file: str | None,
) -> None:
    """Launch the local Sentinel dashboard (FastAPI + uvicorn).

    When ``dashboard.server.require_signed_config: true`` is set, the
    command refuses to start unless ``<config>.sig`` exists and matches
    the resolved config under the supplied signing key.
    """
    try:
        from sentinel.core.exceptions import DashboardNotInstalledError
        from sentinel.dashboard.server import run as run_dashboard
    except ImportError as e:
        raise click.ClickException(
            "Dashboard requires `pip install sentinel-mlops[dashboard]`"
        ) from e

    # Peek at the config first so we know whether signed-config
    # enforcement is enabled. We do this with a non-verifying loader so
    # the error path on bad config is the same regardless of signing.
    try:
        peek_loader = ConfigLoader(config)
        peek_cfg = peek_loader.load()
    except (
        ConfigMissingEnvVarError,
        ConfigCircularInheritanceError,
        ConfigValidationError,
    ) as e:
        raise click.ClickException(str(e)) from e

    if peek_cfg.dashboard.server.require_signed_config:
        keystore = _build_signing_keystore(key_env=config_key_env, key_file=config_key_file)
        try:
            verify_loader = ConfigLoader(config, verify_signature=True, signature_keystore=keystore)
            verify_loader.load()
        except (ConfigSignatureError, AuditKeystoreError) as e:
            raise click.ClickException(
                f"dashboard.server.require_signed_config is true: {e}"
            ) from e

    client = SentinelClient.from_config(config)
    if open_browser:
        import threading
        import time
        import webbrowser

        cfg_host = host or client.config.dashboard.server.host
        cfg_port = port or client.config.dashboard.server.port
        url = f"http://{cfg_host}:{cfg_port}/"

        def _open() -> None:
            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    try:
        run_dashboard(client, host=host, port=port, reload=reload)
    except DashboardNotInstalledError as e:
        raise click.ClickException(str(e)) from e


# ── helpers ────────────────────────────────────────────────────────


def _generate_azure_devops_pipeline(model_name: str, output_dir: Path) -> None:
    """Render the Azure DevOps pipeline template and write it."""
    try:
        import jinja2
    except ImportError as e:
        raise click.ClickException("jinja2 required for --ci: pip install jinja2") from e

    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    template = env.get_template("azure-pipelines.yml.j2")
    rendered = template.render(model_name=model_name)
    dest = output_dir / "azure-pipelines.yml"
    dest.write_text(rendered)
    click.echo(f"wrote {dest}")


def _load_dataset(path: str) -> object:
    """Best-effort dataset loader for the CLI."""
    p = Path(path)
    if p.suffix == ".csv":
        try:
            import pandas as pd  # type: ignore[import-untyped]

            return pd.read_csv(p)
        except ImportError:
            import csv

            with p.open() as f:
                reader = csv.DictReader(f)
                return list(reader)
    if p.suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".parquet":
        try:
            import pandas as pd

            return pd.read_parquet(p)
        except ImportError as e:
            raise click.UsageError("pandas required for parquet support") from e
    raise click.UsageError(f"unsupported file type: {p.suffix}")


# ── shell completion ──────────────────────────────────────────────


@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def shell_completion(shell: str) -> None:
    """Generate shell completion script.

    Example:
        sentinel completion bash >> ~/.bashrc
        sentinel completion zsh >> ~/.zshrc
    """
    scripts = {
        "bash": 'eval "$(_SENTINEL_COMPLETE=bash_source sentinel)"',
        "zsh": 'eval "$(_SENTINEL_COMPLETE=zsh_source sentinel)"',
        "fish": "eval (env _SENTINEL_COMPLETE=fish_source sentinel)",
    }
    click.echo(f"# Add this to your shell profile:\n{scripts[shell]}")


if __name__ == "__main__":
    cli()
