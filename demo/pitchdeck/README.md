# Sentinel pitch deck — module library

This directory holds the slide content for the Sentinel sales pitch as a set of
**self-contained markdown modules**. Each file is one logical slide (sometimes
two). The idea is that you never deliver the whole deck — you pick the modules
that match the customer and assemble a 10–15 slide deck in 5 minutes.

## How to use it

1. Start with `00_title.md` and `10_close.md` — every deck has these.
2. Pick one **problem module** (`01_problem_*.md`) that matches the customer.
3. Pick one **architecture** slide (`02_architecture.md`).
4. Pick 2–4 **capability modules** (`03_*.md` through `08_*.md`) that match
   the customer's actual pain (not everything Sentinel does — just what they care about).
5. Always include `09_compliance.md` if the audience is regulated.
6. Export via your usual tool (Google Slides, Keynote, Marp, Pandoc).

## Standard mixes by customer

| Playbook | Recommended module sequence |
|---|---|
| Bank (A) | `00 → 01_problem_bank → 02_architecture → 03_50line → 04_driftstack → 09_compliance → 10_close` |
| Insurer (B) | `00 → 01_problem_insurer → 02_architecture → 05_llmops → 06_tokenecon → 09_compliance → 10_close` |
| Healthcare (C) | `00 → 01_problem_healthcare → 02_architecture → 07_agentops → 09_compliance → 10_close` |
| E-commerce (D) | `00 → 01_problem_ecommerce → 02_architecture → 08_beyondaccuracy → 10_close` |
| Platform (E) | `00 → 01_problem_platform → 02_architecture → 07_agentops → 03_50line → 10_close` |
| Quant (F) | `00 → 01_problem_quant → 02_architecture → 04_driftstack → 09_compliance → 10_close` |

## Module index

```
00_title.md                 Title + tagline
01_problem_bank.md          UK bank problem framing
01_problem_insurer.md       Insurance modernizer problem framing
01_problem_healthcare.md    Healthcare problem framing
01_problem_ecommerce.md     E-commerce problem framing
01_problem_platform.md      Agent platform team problem framing
01_problem_quant.md         Quant problem framing
02_architecture.md          The seven-layer stack
03_50line_integration.md    From pip install to monitored in 50 lines
04_drift_stack.md           Drift detection that actually works
05_llmops_guardrails.md     LLMOps: guardrails, quality, semantic drift
06_token_economics.md       Token economics + cost predictability
07_agentops_safety.md       AgentOps: tracing, safety rails, tool audit
08_beyond_accuracy.md       Recommendation beyond-accuracy metrics
09_compliance_matrix.md     Compliance framework coverage
10_close.md                 The ask + pilot structure
11_dashboard.md             Local-first browser dashboard (no SaaS)
```

## Adding the dashboard slide

`11_dashboard.md` is an optional capability slide. Drop it in any deck
where the customer wants to *see* something rather than read CLI
output — bank, insurer, healthcare, and platform are the obvious mixes.
Pair with `SENTINEL_DEMO_DASHBOARD=1` so the live demo shows the same
pages the slide previews.
