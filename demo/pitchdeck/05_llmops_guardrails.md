# Slide 05 — LLMOps: guardrails, quality, semantic drift

## Input guardrails (run before any LLM call)

```yaml
guardrails:
  input:
    - type: pii_detection         # Presidio + regex
      action: redact
      entities: [person, ssn, account_number, email, phone]
    - type: jailbreak_detection   # embedding similarity + heuristics
      threshold: 0.85
      action: block
    - type: topic_fence           # semantic scope enforcement
      allowed_topics: [insurance_claims, policy_coverage]
      action: warn
    - type: token_budget          # hard cap before the call
      max_input_tokens: 4000
      action: block
```

## Output guardrails (run after the LLM call)

```yaml
  output:
    - type: toxicity
      threshold: 0.7
      action: block
    - type: groundedness          # NLI / chunk overlap / LLM judge
      method: nli
      min_score: 0.65
    - type: format_compliance     # JSON schema validation
    - type: regulatory_language   # prohibited phrases file
      action: block
```

## Quality + drift on every output

- **LLM-as-judge** scoring on a sample rate (default 10%)
- **Semantic drift** — track embedding centroid of outputs over 7 days
- **Retrieval quality** — relevance, chunk utilisation, faithfulness
- **Prompt drift** — composite signal across quality + violations + tokens

---

*Speaker note:* Call out **PII redaction before the LLM call** as the first
line of defence. The CISO in the room will lean forward. Mention **Presidio**
by name — they've heard of it, it's the Microsoft standard.
