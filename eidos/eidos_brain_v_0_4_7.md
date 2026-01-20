# Eidos Brain v0.4.7.02 — Intended Runtime, Outcomes, Real‑World Use, and Communication Map

This document describes how **EIDOS_BRAIN_UNIFIED_v0_4.7.02** is intended to run end‑to‑end, what it’s supposed to accomplish, how it plugs into real‑world streams (security, markets, general telemetry), and how components communicate—both **inside the engine** and across the broader **Hivemind** deployment.

---

## 1) What Eidos Brain 4.7.02 is (and is not)

### What it is
A **single‑file, unified streaming engine** designed to:

- Ingest a stream of frames (numeric vectors + metadata) from multiple sources.
- Maintain an online predictive model (bicameral: right + left brain).
- Compute **surprise / anomaly** signals via residual‑based gating and secondary monitors.
- Regulate learning (plasticity + thermodynamics) and retain episodic memory (Hippocampus HDC/VSA).
- Emit **forensics artifacts** (top‑K surprises, geometry, checkpoints, reports) suitable for demos and post‑mortem analysis.
- Optionally operate as part of a cloud “Hive” by (a) tailing Pub/Sub/GCS for input and (b) writing artifacts to GCS.

### What it is not
- Not, by itself, a complete trading system, SIEM, or execution layer.
- Not a full microservices deployment: it contains hooks/adapters and storage backends, but the **orchestration** and **event bus contracts** are part of the broader Hivemind layer.

---

## 2) The intended outcomes (what should be accomplished)

Eidos Brain 4.7.02 is intended to produce **four categories of outcomes**:

### A) Online anomaly / surprise detection
- Detect “events that matter” in a stream by measuring **prediction failure** (residual error) and mapping it into **z‑scores** and adaptive thresholds.
- Provide regime/status labels (calibrating, fever, collapse signals, etc.) that are human‑readable and operational.

### B) Bicameral learning + compression
- Maintain a **reservoir‑based right brain** with adaptive readout and a **Newtonian left brain** predictor.
- Convert the stream into a compact **codec** stream (flag + payload) that encodes which frames were “important” and with what fidelity.

### C) Episodic memory and recognition
- Use a Hippocampus (HDC/VSA) to write episodic traces when novelty/surprise is high.
- Enable recognition: if something “looks familiar,” learning can be inhibited/frozen to avoid overfitting and thrashing.

### D) Forensic outputs for demos + debugging
- Produce: top‑K surprise lists, reservoir geometry samples, checkpoints, hippocampus snapshots, and a plain‑language report.
- Support “what happened” explanations through attribution/feature importance on surprise frames.

---

## 3) How it runs (inside the single 4.7.02 process)

### The core loop is always:

**Source Adapter → Frame Generator → Sentinel Stream Loop → Artifact Outputs**

Where:
- **Frame** = numeric vector of fixed dimensionality (`FEATURES`, e.g., 64)
- **Meta** = dict describing provenance (file path, message ID, snippet, timestamps, etc.)

### 3.1 Preflight & environment

On startup, the engine:
- Enforces deterministic CUDA behaviors (when on GPU).
- Sets float precision defaults.
- Resolves artifact roots (Drive or local fallback).
- Initializes the Hive storage backend (Local or GCS) so all outputs have a unified “where artifacts go” abstraction.

### 3.2 Source Adapter modes

The file supports multiple ingestion modes; regardless of mode, the output must become **(frame_vector, meta_dict)**.

**Primary modes (operator‑facing):**
- **DRIVE**: walk a mounted Drive directory or file.
- **LOCAL**: walk local filesystem paths.
- **KAGGLE**: pull dataset and convert numeric columns/rows into frames.
- **STREAM**: live feed (URL line stream, WebSocket, or IP sockets).

**Hive‑native modes (cloud‑facing):**
- **PUBSUB** tail: pull messages from a subscription, vectorize payloads.
- **GCS tail**: scan blobs and convert new content into frames.

### 3.3 The Sentinel stream loop (the “brain”)

Every iteration consumes one `(frame, meta)`:

1) **Normalize / project**
   - Ensure incoming data becomes a fixed `FEATURES` vector.
   - Optional online normalization (Welford) for streams.

2) **Predict**
   - Left brain predicts the next frame via a Newtonian‑style predictor.
   - Right brain predicts via reservoir state and adaptive readout.
   - A consensus (or selected best) prediction is computed.

3) **Compute residual error**
   - `err_L`, `err_R`, `best_err` are calculated.
   - An EMA baseline and sigma estimate maintain running expectations.

4) **Surprise gate**
   - Error becomes a `z_score`.
   - `z_thresh` adapts toward a target surprise rate (`target_surprise`).
   - Determine `is_surprise`.

5) **Hippocampus recognition + write policy**
   - Encode context (reservoir state) and content (frame).
   - Compute similarity and “chi” novelty.
   - Optionally write episodic trace to the appropriate memory bank.
   - Optionally freeze learning when recognized / not novel.

6) **Learning update**
   - If surprise: right brain adapts more aggressively and left brain updates on the true frame.
   - If not surprise: both learn from consensus (or a gentler update) to remain stable.

7) **Secondary monitors**
   - Eigen dominance, spectral entropy/flatness, state entropy.
   - These roll up into human‑readable regime labels.

8) **Record + emit**
   - SessionRecorder appends step telemetry (CSV) and anomalies (JSONL).
   - Top‑K surprises are tracked with meta + attribution.
   - Bicameral compression stream is appended using codec flags:
     - 0: no payload
     - 1: quantized int16 payload
     - 2: raw float32 payload

### 3.4 Finalization

At end (or max frames):
- Prints a summary (frames, surprise rate, thresholds, synaptic hash changes).
- Writes a plain‑language report.
- Saves:
  - reservoir checkpoint
  - hippocampus snapshot
  - geometry artifacts (if enabled)
  - top‑K surprise artifacts
  - compression stream + codec meta

---

## 4) How everything communicates inside the engine

### 4.1 Frame contract (internal)
All ingestion adapters must output:

```text
(vec: np.ndarray[FEATURES], meta: dict)
```

Meta should include at minimum:
- `kind`: one of `row | text | stream | pubsub | bin | ...`
- A provenance field: `path`, `source`, `msg_id`, etc.
- Optional `snippet` for human debugging

### 4.2 Artifact contract (internal)
All outputs go through a single function:

- `store_memory_artifact(data, label, subdir, ext)`

That function:
- writes locally when `HIVE_BACKEND=LOCAL`
- writes through a GCS adapter when `HIVE_BACKEND=GCS`

This ensures the engine can run in Colab, on a laptop, or in Cloud Run/VMs with the same output calls.

### 4.3 Recorder contract
The SessionRecorder is the “truth log” for a run:
- `steps.csv`: step‑level telemetry for charting
- `anomalies.jsonl`: discrete anomaly/surprise events
- `summary.json`: final summary values
- `report.txt`: plain‑language narrative report
- `session_meta.json`: run metadata (profile label, config, session id, roots)

---

## 5) How it is intended to be used in real world events

Eidos Brain is intentionally **domain‑agnostic**: it doesn’t need to know what the stream is; it needs a stable vector and metadata.

### 5.1 Security / operational telemetry
**Goal:** detect attacks, failures, and regime shifts from logs.

Typical sources:
- Suricata EVE JSON, Zeek logs, NGINX JSON, generic NDJSON

Intended pipeline:
- Raw logs → connector → dict events → feature hashing / vectorization → Eidos Brain
- Surprises become alerts + stored forensic packets

### 5.2 Markets / trading telemetry (Kalshi / PolySentinel)
**Goal:** detect tradable novelty and protect execution from instability.

Intended pipeline:
- Market adaptor ingests:
  - order books / trades / prices / implied probabilities
  - news / event updates (optional)
- Adapt into frames:
  - numeric deltas, spreads, volatility features, time‑of‑day encodings
- Eidos Brain outputs:
  - surprise / regime label
  - confidence / attribution
  - memory recognition (“seen this pattern before?”)
- Execution layer uses outputs:
  - **risk gates**: don’t trade in collapse/fever
  - **intent generation**: only trade when novelty is high and signal is consistent
  - **position sizing**: use regime + z energy

### 5.3 “What’s worth saying” (roadmap goal)
**Goal:** run as a cognitive filter over any firehose.

Intended pipeline:
- Feeds: web streams, social feeds, RSS, sensor data
- Output: a reduced stream of “high‑surprise” frames + narrative report

This becomes the basis of:
- “Eidos Live” style anomaly storytelling
- Investor demos showing *why* it flagged what it flagged

---

## 6) Hivemind (cloud) communication architecture

This is the deployment‑level communication layer that the project conversations describe.

### 6.1 The central rule
**Everything becomes an event** on a shared bus, and every agent is a subscriber/publisher.

### 6.2 Canonical event envelope (recommended)
All producers (trader, sentinels, reconcilers, advancers) publish JSON events of the form:

```json
{
  "ts": "2026-01-19T00:00:00Z",
  "id": "uuid",
  "domain": "market|security|system",
  "agent_id": "kalshi-trader|hive-sentinel|hive-reconciler|hive-advancer",
  "kind": "tick|order_intent|fill|anomaly|checkpoint|config_update",
  "data": { "...": "payload" },
  "features": [0.1, 0.2, "..."],
  "meta": { "source": "...", "snippet": "..." }
}
```

### 6.3 Service roles (intended)
- **Trader / Executor** (`kalshi-eidos-trader`): market adaptor + Eidos Brain + risk gates + order execution.
- **Ingestor** (`hive-ingestor`): Pub/Sub → BigQuery as system of record.
- **Sentinel** (`hive-sentinel` / PolySentinel): parallel subscriber that audits anomalies/regimes and can issue stop actions.
- **Reconciler** (`hive-reconciler`): positions/fills/PnL truth.
- **Advancer** (`hive-advancer`): evaluates outcomes and emits updated configs / snapshots.

### 6.4 How it maps to 4.7.02 code
The 4.7.02 file already contains:
- A **Pub/Sub input tail** generator (for pulling events)
- A **GCS artifact store** (for writing outputs)

The missing piece for “full hivemind” is not the math; it’s the production wrappers:
- stable schema enforcement
- hot config reload
- durable state for executions/positions
- explicit publish of every stage back into the bus

---

## 7) Operator view: how to run it (intended)

### 7.1 Local / Drive / Kaggle research run
- Set `DATA_SOURCE_TYPE` to `LOCAL`, `DRIVE`, or `KAGGLE`.
- Set dataset/source paths.
- Set `FEATURES` and `EIDOS_BRAIN_CONFIG`.
- Run `run_eidos_sentinel()`.

### 7.2 Live stream run
- Set `DATA_SOURCE_TYPE="STREAM"`.
- Choose `STREAM_KIND` and configure `STREAM_URL` or `STREAM_IP_ENDPOINT`.
- Ensure parsing mode (`STREAM_EVENT_FORMAT`, `STREAM_SECURITY_FEATURIZE`) matches the stream.
- Run `run_eidos_sentinel()`.

### 7.3 Cloud/Hive run
- Use `HIVE_BACKEND="GCS"` so outputs land in buckets.
- Use the Pub/Sub tail mode as the generator source (or wrap it via NL connector injection).
- Deploy as:
  - Cloud Run service (always on)
  - or VM/GKE job (for long live feeds)

---

## 8) What “communication” means at every layer

### Inside one process
- Ingestion adapters communicate via the `(vec, meta)` contract.
- The brain communicates via shared state: reservoir, left predictor, hippocampus.
- Recorder communicates by appending telemetry and anomaly events.
- Artifacts communicate by writing stable files (local or GCS).

### Between processes (Hivemind)
- Everything communicates via:
  - Pub/Sub (real‑time bus)
  - BigQuery (system of record)
  - GCS (artifacts/snapshots/config)

### Human/operator communication
- Plain‑language report explains what was seen.
- Top‑K surprise text file gives a quick “what and where.”
- Geometry + checkpoint artifacts support visual demos.

---

## 9) Practical “real world” integration patterns (how you utilize it)

### Pattern A — Streaming anomaly sentinel (security)
1) Collector → Pub/Sub topic `hive-events` (or direct stream)
2) Eidos Brain reads Pub/Sub subscription, emits anomalies
3) Alerting service triggers on anomalies
4) BigQuery stores everything for dashboards

### Pattern B — Trading system (adapter → brain → execution)
1) Market adaptor builds features for each market tick
2) Eidos Brain returns:
   - z_score / surprise, regime, attribution
3) Risk gate determines if allowed to trade
4) Execution places order
5) Fills/outcomes published back to bus
6) Reconciler and advancer close the learning loop

### Pattern C — Investor/demo mode
- Run Kaggle or archive mode on a known dataset.
- Show:
  - surprise spikes
  - regime labeling
  - geometry artifact visuals
  - checkpoint evolution
  - compression ratio

---

## 10) Open gaps (known) to make this fully “customer ready”

These are the non‑math gaps that turn the engine into a product:

1) **Adapters** per domain that reliably produce numeric frames.
2) **Event schema** enforcement + versioning.
3) **Hot reload** configs/checkpoints from GCS (and safe rollout).
4) **Secret handling** (no hardcoded keys; use Secret Manager / env).
5) **Durable state** for trading/positions (Firestore/Redis/postgres).
6) **Ops**: health checks, structured logs, metrics, alarms.

---

## 11) One‑screen mental model

```text
       (ANY REAL WORLD SOURCE)
   logs / markets / web / sensors
              |
              v
        [ ADAPTER LAYER ]
  normalize + featurize + meta
              |
              v
     (vec[FEATURES], meta)
              |
              v
      [ EIDOS BRAIN 4.7.02 ]
  bicameral prediction + surprise gate
  hippocampus memory + regime monitors
              |
              +-------------------+
              |                   |
              v                   v
     [ ACTION / EXEC ]      [ ARTIFACTS / RECORD ]
  trade / alert / stop     top-k / report / ckpt
              |
              v
        [ HIVE EVENT BUS ]  (Pub/Sub)
              |
              v
        [ SYSTEM OF RECORD ] (BigQuery)
```

