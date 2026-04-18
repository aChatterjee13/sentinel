# Project Sentinel — demo kit

Everything you need to run a 15–60 minute Sentinel demo for a prospective
customer. No cloud dependencies. Every script runs offline with synthetic data.

## Directory layout

```
demo/
├── README.md                   ← you are here
├── configs/                    ← 6 rebadged config files, one per playbook
│   ├── credit_decisioning.yaml       (Playbook A — UK bank)
│   ├── claims_rag.yaml               (Playbook B — Insurance modernizer)
│   ├── clinical_decision.yaml        (Playbook C — Healthcare)
│   ├── product_reco.yaml             (Playbook D — E-commerce)
│   ├── multi_team_agents.yaml        (Playbook E — Agent platform team)
│   └── volatility_forecast.yaml      (Playbook F — Quant)
├── envs/                       ← sourceable env files per playbook
├── data/                       ← synthetic data generators
├── agents/                     ← stand-in agents for AgentOps demos
├── scripts/                    ← 5-act demo runners, one per playbook
├── leavebehind/                ← 1-page markdown handouts per playbook
└── pitchdeck/                  ← remix-able pitch deck modules
```

## The six playbooks

| # | Playbook | Customer profile | Lead config |
|---|---|---|---|
| A | `bank` | UK regulated retail bank, credit/fraud models | `credit_decisioning.yaml` |
| B | `insurer` | Insurance modernizer, tabular + RAG | `claims_rag.yaml` |
| C | `healthcare` | Provider with clinical decision support | `clinical_decision.yaml` |
| D | `ecommerce` | Growth-focused retailer, recommender systems | `product_reco.yaml` |
| E | `platform` | Enterprise agent platform team | `multi_team_agents.yaml` |
| F | `quant` | Asset manager / quant, forecasting | `volatility_forecast.yaml` |

## Quick start — run a demo in 60 seconds

Pick the playbook that matches the customer. From the repo root:

```bash
# Example: UK bank
source demo/envs/bank.env
bash demo/scripts/bank_demo.sh
```

The script will:

1. Generate synthetic data for that playbook (no real data, ever)
2. Validate the config
3. Walk through 5–7 acts, pausing between each so you can talk
4. End with talking-point callouts you can read verbatim

Repeat with:

```bash
source demo/envs/insurer.env     && bash demo/scripts/insurer_demo.sh
source demo/envs/healthcare.env  && bash demo/scripts/healthcare_demo.sh
source demo/envs/ecommerce.env   && bash demo/scripts/ecommerce_demo.sh
source demo/envs/platform.env    && bash demo/scripts/platform_demo.sh
source demo/envs/quant.env       && bash demo/scripts/quant_demo.sh
```

## What each demo proves

### Playbook A — `bank_demo.sh`
PSI drift on a credit decisioning model. Clean window passes, drifted
window (cost-of-living shock) fires. 40-line config delivers FCA + PRA
compliance. Auto-rollback canary wired in. 7-year audit retention default.

### Playbook B — `insurer_demo.sh`
RAG guardrails: PII redaction in flight, jailbreak blocking, topic fence,
token budget enforcement, groundedness check. Same audit trail shape as
the tabular fraud model. Cost-per-query hard cap at $0.50.

### Playbook C — `healthcare_demo.sh`
Clinical decision support agent with PHI hashing, confidence-based
escalation (agent hands off at 32% confidence), EHR-write blocked by tool
permissions, 7-year audit retention. HIPAA + SaMD compliance pack.

### Playbook D — `ecommerce_demo.sh`
Recommendation model with long-tail collapse detection. Clean window passes
all beyond-accuracy metrics; collapsed window fires coverage + popularity
bias + fairness alerts. Revenue-aware canary rollback.

### Playbook E — `platform_demo.sh`
LangGraph + Semantic Kernel agents under one Sentinel config. Normal tasks
succeed. Buggy loop agent is killed by the safety rail. Tool permission
enforcement prevents an email from being sent. Per-team budgets tracked.

### Playbook F — `quant_demo.sh`
Equity volatility forecast with calendar-aware drift (same-period vs
same-period). Clean forward month passes, regime-shift month triggers
trend-slope alert + ADF stationarity loss + calendar drift. SR 11-7
lineage pack. Shadow-first deployment with 14-day paper-trading parity.

## Customizing for a specific customer

1. **Rename the model**: edit `model.name` in the relevant config.
2. **Add their logo / colour**: the scripts use plain ANSI escapes, override
   `SAY` in the script if you want.
3. **Point at their data**: replace the `--reference` / `--current` arguments
   in the script with their file paths.
4. **Turn on real alerts**: set `SLACK_WEBHOOK_URL` in the env file before
   sourcing it. The demo will fire real alerts.
5. **Pick pitch deck modules**: see `demo/pitchdeck/README.md` for
   playbook-to-module mappings.
6. **Hand over the leave-behind**: print `demo/leavebehind/<playbook>.md`
   as a PDF and leave it with the customer.

## Prerequisites

- Python 3.10+ (CPython 3.10, 3.11, 3.12, or 3.13 all work)
- `pip install -e ".[all]"` (one-time, from repo root) — brings in scipy,
  sklearn, plotly, shap, tiktoken, openai, opentelemetry, statsmodels,
  networkx, etc. Intentionally *does not* pull torch /
  sentence-transformers / presidio so the install resolves on Python 3.13
  (torch has no wheels there yet) and lightweight CI images.
- For the `bank` and `quant` playbooks: pandas + pyarrow (part of `[all]`)
- For the `insurer`, `healthcare`, `platform` playbooks: everything runs
  in mock LLM mode by default, no API keys required.
- *Optional, only if you actually want embedding-based semantic drift,
  jailbreak detection, topic fencing, or PII analysis:*
  `pip install -e ".[all,dashboard,ml-extras]"` — adds
  sentence-transformers, presidio, and spacy. Requires Python ≤3.12 on
  most platforms because of torch wheel availability.
- *Optional but highly recommended for live demos:*
  `pip install -e ".[all,dashboard]"` — adds FastAPI + Jinja2 + uvicorn
  for the live dashboard described below.

## Showing the dashboard alongside any playbook

Every demo script can optionally launch the Sentinel dashboard in the
background while the terminal narration runs. The dashboard is the same
data the CLI is printing — just rendered as a browser UI with charts,
filters, and per-page deep links — so the prospect can follow along
visually instead of reading raw stdout.

Enable it by exporting `SENTINEL_DEMO_DASHBOARD=1` before running any
script:

```bash
pip install -e ".[all,dashboard]"
export SENTINEL_DEMO_DASHBOARD=1
source demo/envs/bank.env
bash demo/scripts/bank_demo.sh
```

What happens:

1. The script boots `sentinel dashboard --config $CONFIG --port 8000` in
   the background (logs to `/tmp/sentinel-dashboard.log`).
2. It pauses once at the top so you can open
   `http://127.0.0.1:8000` in a browser and arrange windows next to the
   terminal.
3. As each act fires drift checks, registers models, records audit
   events, etc., the dashboard pages refresh with the new data.
4. When the script exits, the dashboard process is killed automatically
   via a `trap`.

Override the port with `SENTINEL_DEMO_DASHBOARD_PORT` (default `8000`)
if `8000` is already in use locally.

The dashboard is read-only over the live `SentinelClient` — there is no
extra database to provision and nothing to keep in sync, so even when
the script tears everything down at the end the demo state stays
self-contained on disk under `./audit/` and `./registry/`.

## No real data. No real PII. No real PHI.

Every dataset in this kit is generated from numpy. Every query in the
insurance / healthcare demos is synthetic. The demo kit is safe to clone
onto a customer's laptop without any DPIA concerns.

## Rehearsal tips

- Run every demo twice before delivering it live. Know where the pauses land.
- Read the talking-point block at the end of each script — those are the
  lines you want to deliver verbatim at the close.
- Start with the **config file on screen, not the architecture diagram**.
  Configs feel concrete; diagrams feel like vapourware.
- Leave the last 10 minutes for *their* data. The single strongest close
  is pointing Sentinel at one of *their* files, live, and showing it work.
