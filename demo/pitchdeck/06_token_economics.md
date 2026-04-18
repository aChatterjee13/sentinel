# Slide 06 — Token economics: cost predictability at last

## The config

```yaml
token_economics:
  track_by: [model, prompt_version, user_segment, use_case]
  budgets:
    daily_max_cost: 500.00
    per_query_max_tokens: 8000
    per_query_max_cost: 0.50
  alerts:
    daily_cost_threshold: 400.00
    cost_per_query_trend_increase_pct: 20
```

## What you get

- **Attribution**: every LLM call tagged with model, prompt version, user
  segment, and use case. Slice the bill any way.
- **Hard budgets**: per query, per day, per tenant. Blocked at SDK level,
  not at the cloud bill level.
- **Trend alerts**: cost-per-query up 20% week-on-week → alert, because
  that's what a prompt regression looks like before it shows up in quality.
- **Model routing logs**: which model was chosen and why — so you can
  decide when to downgrade to a cheaper model.
- **Custom pricing tables**: works with Azure OpenAI, OpenAI direct,
  Anthropic, open-source on vLLM. Not locked to one provider's rate card.

## The single sentence your CFO wants to hear

> "Our LLM cost is bounded, attributed, and alerted on. We cannot have a
> surprise bill."

---

*Speaker note:* For the insurance playbook, tie this directly back to the
**"$4,000 bad prompt"** line from slide 01. Circular storytelling works.
