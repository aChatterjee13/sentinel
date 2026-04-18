# Sentinel for E-commerce Growth Teams

**NDCG isn't enough. Catch the silent killers — long-tail collapse, popularity bias, fairness gaps — before they eat your revenue.**

---

## The problem we heard from you

- Your recommender gets re-trained nightly but nobody knows if the
  re-training actually improved anything that matters to the business.
- Your vendor dashboard shows NDCG@10 going up. Your revenue per session
  is going down. You suspect the model is collapsing onto a narrow item set.
- A new cohort of customers is getting worse recommendations than your loyal
  segment. You don't know how to measure that defensibly.
- Every canary release is a gut call. Rollbacks are manual. The last one
  cost you £120k in lost sales before anyone noticed.

## What Sentinel does about it

| Metric | Why it matters |
|---|---|
| Coverage (% of catalogue recommended) | Long-tail collapse detection |
| Diversity (intra-list similarity) | Are recommendations repetitive? |
| Novelty (inverse popularity) | Are we showing things users can find themselves? |
| Popularity bias (Gini) | Concentration on top-1% of items |
| Group fairness (NDCG gap by segment) | Regulator defensibility |
| Cold-start ratio | New product surfacing health |
| Revenue-aware canary rollback | `rollback_on: { revenue_drop_pct: 0.03 }` |
| A/B test guardrail integration | `guardrail_metrics: [revenue_per_session, engagement_rate]` |

## What you saw in the demo

1. A baseline recommendation window passed all metrics cleanly.
2. A **collapsed window** (all recs from top-1% items) triggered:
   - Coverage alert: dropped from 48% to 1%
   - Popularity bias alert: Gini jumped from 0.62 to 0.98
   - Long-tail ratio alert
3. The model would have been **auto-rolled back** before it ever reached 100%
   traffic because the revenue guardrail also fired.

## Commercial headline

- **Beyond-accuracy metrics on day 1**: Coverage, diversity, novelty,
  popularity bias, serendipity, fairness. Your vendor ships none of these.
- **Revenue-aware canary**: Roll back on revenue per session, not just on
  model error. The business KPI is the circuit breaker.
- **Regulator-ready fairness**: Track NDCG gap across user segments. Prove
  parity on demand.
- **Full A/B test integration**: Optimizely / LaunchDarkly hooks built in.

## What we need from you to run a pilot

1. One recommendation surface (homepage, category page, cart add-on).
2. Read-only access to your interaction logs (or a synthetic export).
3. Your current production vendor metrics — so we can A/B Sentinel alongside.
4. 2 weeks of part-time effort from one data scientist + one platform engineer.
5. At the end: a dashboard showing 6 months of backfilled beyond-accuracy
   metrics, plus one caught "silent failure" you can bring to your product team.

## The ask

A 45-minute session with your head of growth + head of data science. We
backfill your last 90 days of recommendations through Sentinel live and
show you at least one issue you didn't know you had.

---

*Contact:* your Zensar account team • *Demo repo:* `demo/scripts/ecommerce_demo.sh`
