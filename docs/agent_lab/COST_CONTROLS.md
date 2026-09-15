# Cost Controls

Routing order is deterministic Python, cached project evidence, Archivist/Luna, Bench/Mini, Terra,
Codex, Sol, then Astra. This is policy, not a claim that the cheapest model is always sufficient.
Director escalates only when the lower layer cannot reliably answer the question.

`BudgetManager` records calls per model/agent, tool calls, input tokens, cached input tokens, output
tokens, reasoning tokens when reported, retries/turn limits and Council use. It enforces configured
task and per-agent dollar limits plus specialist-call caps. Council is disabled by default.

Prices live in `config/agent_lab/model_pricing.yaml`, separate from routing. The checked-in registry has
`null` prices because current reviewed API billing was not established by this build. Receipts therefore
label live cost unknown instead of inventing an exact amount. Add reviewed per-million-token prices,
change the version/date, and document tool charges before relying on estimates.

No universal dollar budget is baked in. Configure:

```text
EIDOS_AGENT_TASK_BUDGET_USD=
EIDOS_AGENT_HUMAN_APPROVAL_THRESHOLD_USD=
EIDOS_AGENT_MAX_SPECIALIST_CALLS=12
EIDOS_AGENT_MAX_RETRIES=1
EIDOS_AGENT_MAX_TURNS=16
```

Model substitution is explicit. If a configured model is unavailable, execution fails unless an
`EIDOS_AGENT_<ROLE>_FALLBACK_MODEL` is configured and available. Every substitution enters the run
manifest.
