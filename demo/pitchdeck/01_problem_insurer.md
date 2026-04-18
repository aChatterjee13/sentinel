# Slide 01 — The problem (insurance modernizer)

## Your fraud model is governed. Your RAG pipeline isn't.

| Model type | Governed today? | Why not |
|---|---|---|
| Tabular fraud | ✅ | You spent 5 years building it |
| Claims RAG | ❌ | "LLMs are different" |
| Policy Q&A chatbot | ❌ | A different team runs it |
| Underwriter agent | ❌ | It's a pilot, we'll sort it later |

### What "not governed" actually means

- **PII is in your LLM provider's logs right now.** Your DPO knows.
- **Token cost is unpredictable.** One bad prompt ate $4,000 last month.
- **Hallucinations are found by customers, not by you.**
- **The compliance committee is asking for evidence you don't have.**

---

*Speaker note:* The PII line is the attention-getter. Deliver it slowly. Let
them sit with it for 3 seconds before moving on.
