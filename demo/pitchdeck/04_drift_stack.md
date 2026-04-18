# Slide 04 — Drift detection that actually works

## Three independent layers — configured, not coded

### Data drift (input distribution)

| Method | Best for |
|---|---|
| PSI | Binned continuous + categorical |
| KS | Continuous features |
| Jensen-Shannon | Probability distributions |
| Chi-squared | Categorical |
| Wasserstein | Shape-sensitive continuous |

### Concept drift (X → y relationship)

| Method | Best for |
|---|---|
| DDM | Fast-label binary classification |
| EDDM | Gradual drift |
| ADWIN | Streaming with adaptive window |
| Page-Hinkley | Mean shift in error rate |

### Model drift (performance decay)

Tracked metrics (any): accuracy, F1, AUC, MAPE, MASE, coverage, NDCG,
Hits@K, MRR, modularity...

### Plus — domain-aware drift

- **Time series:** calendar-aware tests, STL decomposition, ADF stationarity
- **NLP:** vocabulary drift, embedding space MMD, label distribution
- **Recommendation:** item/user distribution, long-tail ratio, cold-start
- **Graph:** topology drift, degree distribution, clustering coefficient
- **LLM:** semantic output drift via embedding centroids

---

*Speaker note:* The single line: *"One config field controls everything.
`method: psi` → `method: calendar_test` is all it takes to switch from
tabular to time series monitoring."*
