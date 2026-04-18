# Sentinel for Insurance Modernizers

**Govern tabular models and LLM pipelines with the same SDK, the same audit trail, the same compliance framework.**

---

## The problem we heard from you

- Your fraud models are under model risk governance. Your claims RAG
  pipeline is not — because there is no tool that does both.
- PII is going to an external LLM provider's logs. Your CISO knows it.
  Your DPO knows it. Compliance can see the cliff.
- Token bills are unpredictable. One bad prompt ate $4,000 last month.
- Hallucinations are being caught by customers, not by you.
- Two separate teams are building two separate "monitoring" stacks because
  "LLMs are different".

## What Sentinel does about it

| Capability | How Sentinel delivers |
|---|---|
| PII redaction before LLM call | `guardrails.input: [{ type: pii_detection, action: redact }]` |
| Jailbreak detection | `jailbreak_detection: { method: embedding_similarity }` |
| Groundedness / hallucination | `groundedness: { method: nli, min_score: 0.65 }` |
| Topic fence | `topic_fence: { allowed_topics: [insurance_claims, policy_coverage] }` |
| Token budgets | `daily_max_cost: 500.00`, `per_query_max_cost: 0.50` |
| Prompt versioning + A/B | `prompts.versioning: semantic, ab_testing.default_split: [90, 10]` |
| Semantic output drift | `semantic_drift.threshold: 0.15` over 7-day window |
| Same audit trail as fraud | One `audit:` block, FCA + GDPR + EU AI Act frameworks |

## What you saw in the demo

1. A PII-laden query hit the guardrail pipeline. It was **redacted before
   the LLM provider ever saw it**.
2. A jailbreak query was **blocked**. The audit trail has the full reason.
3. An off-topic query was **warned**. Traffic is still served, but flagged.
4. The groundedness check runs on every output. No hallucination slips through.
5. The token economics dashboard shows cost per prompt version, per user
   segment, per use case.

## Commercial headline

- **One governance framework** for tabular, NLP, and LLM models.
- **PII never reaches your LLM provider**. Defensible to the DPO on day 1.
- **Cost predictability**: Hard budgets per query, per day, per user segment.
- **Regulator-ready**: EU AI Act high-risk classification, full audit lineage,
  prompt version history, guardrail violation log.
- **Framework agnostic**: OpenAI, Azure OpenAI, Anthropic, open-source on vLLM.

## What we need from you to run a pilot

1. One RAG application (claims Q&A, policy lookup, underwriter assistant).
2. Access to whichever LLM provider you're using.
3. A list of the phrases / patterns you currently consider prohibited.
4. 2 weeks of part-time effort from one ML engineer and one compliance lead.
5. At the end: a full guardrail + cost + quality dashboard on one pilot
   pipeline, plus the compliance pack for your model risk committee.

## The ask

45 minutes with your CISO / DPO to walk through the PII redaction guarantee
and show the audit trail of a demo PII query being caught. We bring the
demo; you bring the toughest question you have.

---

*Contact:* your Zensar account team • *Demo repo:* `demo/scripts/insurer_demo.sh`
