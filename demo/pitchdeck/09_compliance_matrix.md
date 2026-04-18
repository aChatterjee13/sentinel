# Slide 09 — Compliance framework coverage

## Supported out of the box

| Framework | Jurisdiction | Sentinel coverage |
|---|---|---|
| **FCA Consumer Duty** | UK | Outcome monitoring, fairness, full lineage, 7yr retention |
| **PRA SS1/23** | UK | Model risk lifecycle, validation gates, approval chains |
| **EU AI Act** | EU | Risk classification, transparency docs, human oversight |
| **GDPR** | EU | PII redaction, right-to-erasure hooks, audit lineage |
| **HIPAA** | US | PHI redaction/hash, access audit, 7yr retention |
| **FDA SaMD** | US | Clinical decision traceability, model version history |
| **SR 11-7** | US (banks) | Model risk management, champion-challenger, validation |
| **SOC 2** | Global | Access logs, change management, audit immutability |
| **ISO/IEC 42001** | Global | AI management system documentation |

## The magic line in the config

```yaml
audit:
  compliance_frameworks: [fca_consumer_duty, pra_ss1_23, eu_ai_act]
```

This single line:

- Sets the default retention (2555 days for UK banks)
- Enables the compliance report generator for each framework
- Tags every audit event with the framework(s) it satisfies
- Unlocks one-command evidence pack export

## What it's not

It is **not** a substitute for your compliance team. It is the **evidence
layer** your compliance team needs — ready from day one instead of from
month six.

---

*Speaker note:* Name the framework that matters to this customer first.
Don't list all nine — pick the 2–3 they care about and skip the rest.
