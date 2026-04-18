# Slide 02 — Architecture

## The seven-layer stack

```
┌──────────────────────────────────────────────────────┐
│  1. Developer Interface                              │
│     SDK Core API │ Config as Code │ CLI             │
├──────────────────────────────────────────────────────┤
│  2. Observability                                    │
│     Data Quality │ Drift │ Feature Health │ Cost    │
├──────────────────────────────────────────────────────┤
│  3. LLMOps                                           │
│     Prompts │ Guardrails │ Quality │ Token Economics│
├──────────────────────────────────────────────────────┤
│  4. AgentOps                                         │
│     Tracing │ Tool Audit │ Safety │ Agent Registry  │
├──────────────────────────────────────────────────────┤
│  5. Intelligence                                     │
│     Model Graph │ KPI Link │ Explainability         │
├──────────────────────────────────────────────────────┤
│  6. Action                                           │
│     Notifications │ Retrain │ Deployment            │
├──────────────────────────────────────────────────────┤
│  7. Foundation                                       │
│     Registry │ Audit Trail │ Experiments            │
└──────────────────────────────────────────────────────┘
```

### The critical thing

Every layer feeds the same **audit trail** and the same **notification
engine**. A drift alert from layer 2, a guardrail violation from layer 3,
and a safety rail firing from layer 4 all flow through the same pipeline and
land in the same compliance report.

---

*Speaker note:* Don't walk through every layer. Point at 2 / 3 / 4 and say
*"tabular, LLM, agents — one stack"*. That's the only sentence that matters.
