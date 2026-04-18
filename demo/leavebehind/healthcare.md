# Sentinel for Healthcare

**Agents in clinical contexts — with hard safety rails, clinician-in-the-loop by default, and a medical-legal audit trail.**

---

## The problem we heard from you

- Every clinician on your team has the same question: *how will we know
  when the AI is wrong?*
- HIPAA makes every third-party tool a compliance review.
- You cannot let an agent write to the EHR. Ever.
- The medical-legal team wants the full reasoning trace for any case where
  the AI influenced a decision.
- You need to run clinical decision support, drug interaction checks, and
  evidence retrieval — but nothing that prescribes, orders, or escalates
  without a human.

## What Sentinel does about it

| Need | How Sentinel delivers |
|---|---|
| PHI never leaves your tenant | `pii_detection: { action: redact, strategy: hash }` |
| No EHR writes, ever | `tool_audit.permissions: { blocked: [ehr_write, prescription_write] }` |
| Hard loop ceiling | `loop_detection.max_iterations: 30` (tighter than default) |
| Every reasoning step traced | `tracing.sample_rate: 1.0`, `retention_days: 2555` |
| Clinician-in-the-loop below 70% confidence | `escalation.triggers: [{ confidence_below: 0.70, action: human_handoff }]` |
| Unanimous multi-agent consensus | `multi_agent.consensus.min_agreement: 1.0` |
| Sandboxed destructive ops | `sandbox.mode: approve_first` — prescribe/order always gated |
| SaMD + HIPAA audit package | `compliance_frameworks: [hipaa, eu_ai_act, fda_samd]` |

## What you saw in the demo

1. A clean clinical case was processed end-to-end. Every tool call traced.
2. A **low-confidence case was caught at 32% confidence** and escalated to a
   clinician *automatically* — before any recommendation was generated.
3. A case with PHI in the summary had the PHI **hashed before any LLM call**.
4. The audit trail shows the full span tree for every run, 7-year retention.

## Commercial headline

- **Clinician-in-the-loop is a config line, not a refactor.**
- **Your EHR is untouchable by the agent.** Tool permissions enforced at SDK level.
- **Medical-legal-ready**: Full reasoning trace per run, stored for 7 years.
- **Framework agnostic**: Works with LangGraph, Semantic Kernel, Google ADK.
- **HIPAA aligned**: PHI never sent to third-party LLM providers by default.

## What we need from you to run a pilot

1. One non-diagnostic use case (triage, evidence retrieval, care pathway lookup).
2. Access to a test environment with synthetic patient data only.
3. 2 weeks with one ML engineer + 1 clinician for the working sessions.
4. Clear escalation pattern: who gets paged when confidence drops?
5. At the end: a working clinical decision support agent with the full
   safety envelope on, ready to show your medical-legal team.

## The ask

30 minutes with your chief medical informatics officer and your
medical-legal counsel. We walk through the trace-replay capability and the
SaMD compliance pack. They tell us what would need to be true for this to
go into a pilot.

---

*Contact:* your Zensar account team • *Demo repo:* `demo/scripts/healthcare_demo.sh`
