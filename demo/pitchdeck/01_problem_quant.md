# Slide 01 — The problem (quant / asset manager)

## Why your drift tool screams every January

Standard drift detectors (PSI, KS, JS) assume independence between
observations. Time series laughs at that assumption.

| Pattern | Standard drift tool says | Reality |
|---|---|---|
| January 2026 vs July 2025 baseline | DRIFT — investigate | Normal seasonality |
| 2026 Q1 earnings season | DRIFT — investigate | Expected vol spike |
| Weekend gap in hourly data | DRIFT — investigate | Trading calendar |
| Actual regime change | *buried in the noise of false positives* | ← the one you need |

### And SR 11-7 compliance

- Full lineage: built by hand at audit time. **2 weeks per model.**
- Per-horizon skill: tracked in a spreadsheet. **Nobody trusts it.**
- Champion-challenger history: reconstructed from git commits. **Embarrassing.**

---

*Speaker note:* The quant reaction you're looking for: *"You actually know
what calendar-aware drift means."* That's the moment you've earned the next
30 minutes.
