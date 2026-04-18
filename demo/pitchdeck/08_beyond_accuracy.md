# Slide 08 — Recommendation beyond-accuracy metrics

## Your vendor gives you NDCG. Sentinel gives you the rest.

| Metric | What it measures | Why it matters |
|---|---|---|
| **Coverage** | % of catalogue recommended | Long-tail collapse detection |
| **Diversity (ILS)** | Intra-list similarity | Are recs repetitive? |
| **Novelty** | Inverse popularity | Surfacing new things? |
| **Serendipity** | Unexpected + relevant | The holy grail |
| **Popularity bias (Gini)** | Concentration on top items | Are you just showing top-1%? |
| **Fairness (group)** | NDCG parity across segments | Regulator-defensible |
| **Cold-start ratio** | New items getting impressions | Catalogue health |

## The single yaml block

```yaml
domains:
  recommendation:
    quality:
      beyond_accuracy:
        coverage:       { min_threshold: 0.4 }
        diversity:      { min_threshold: 0.3 }
        popularity_bias:{ max_threshold: 0.8 }
      fairness:
        protected_attribute: user_segment
        metric: ndcg_at_10
        max_disparity: 0.10
```

## Revenue-aware canary rollback

```yaml
deployment:
  strategy: canary
  canary:
    rollback_on:
      revenue_drop_pct: 0.03
      engagement_drop_pct: 0.05
```

The business KPI is the circuit breaker, not just model error.

---

*Speaker note:* If they're using Amazon Personalize, Algolia Reco, or
something similar — this is the slide that makes them wince.
