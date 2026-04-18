# Slide 03 — From `pip install` to monitored in 50 lines

```python
from fastapi import FastAPI
from sentinel import SentinelClient
import joblib

app = FastAPI()
model = joblib.load("models/credit.pkl")
sentinel = SentinelClient.from_config("sentinel.yaml")

@app.on_event("startup")
async def startup():
    sentinel.register_model_if_new(version="3.2.1", framework="xgboost")

@app.post("/predict")
async def predict(features: dict):
    quality = sentinel.check_data_quality(features)
    if quality.has_critical_issues:
        return {"error": "Input validation failed"}
    prediction = model.predict([features])
    sentinel.log_prediction(features=features, prediction=prediction)
    return {"prediction": prediction.tolist()}

@app.get("/health")
async def health():
    return {
        "model": sentinel.model_name,
        "drift": sentinel.check_drift().summary,
        "feature_health": sentinel.get_feature_health().summary,
    }
```

### What you get for free

- Schema validation on every input
- Rolling drift window (PSI/KS/JS — whichever you configured)
- Registered model version with baseline metrics
- Audit trail entry for every prediction
- Slack/Teams/PagerDuty alerts wired to your escalation chain
- Canary deployment ready via `sentinel deploy`

---

*Speaker note:* Deliver the line **"that's everything"** after you read the
health endpoint. Then pause. The pause is the pitch.
