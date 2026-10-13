# Sentinel Observatory

**Live URL/IP upgrade:** [Connect a target and watch](live/README.md) adds a
hosted live URL monitor and a shared local companion for URLs/public IPs. The
original fixed-route implementation below remains preserved.

A local, observational vertical slice for the **public HTTP edge of Eidos Works**.
The evidence field shows actual measurements, opens their original records, and
separates measured facts, experimental change signals, numerical forecasts and
unverified language-model interpretations.

This is not a packet monitor, intrusion detector, WAF, or proven security product.
It makes no protective changes and cannot establish that a site is secure.

## Run

From the repository root, with Python 3.11 or newer:

```sh
python -m pip install -r apps/sentinel-observatory/requirements.txt
cd apps/sentinel-observatory
python server.py
```

Open `http://127.0.0.1:8765`. The service binds loopback only. It makes one HEAD
request per route at a requested minimum interval of 30 seconds, sequentially;
slow requests extend the actual sampling interval. The UI polls independently.
Stop with Ctrl-C. Nothing installs a daemon or adds credentials.

The four fixed targets are the homepage, account shell, Community and Playground
on `https://eidos-works.com`. The diagnostic button queues a single HEAD request
to a deliberately missing fixed path. It records the **actual** response and
labels the request as a controlled diagnostic, never an attack. Repeated probes
are limited to one per 30 seconds.

```sh
python -m unittest discover -s . -p 'test_*.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Optional local interpreter

The tested model adapter uses official `Qwen/Qwen3-0.6B-GGUF` Q8_0 weights through
`llama-cpp-python`. Install that optional runtime and download the weights only
when you want local interpretation, then configure:

```sh
export OBSERVATORY_MODEL_PATH=/absolute/path/Qwen3-0.6B-Q8_0.gguf
python server.py
```

No hosted paid AI API is called. Weights are not part of this repository. The
`Interpreter` interface is replaceable; the built-in prompt adapter is specifically
for Qwen3 ChatML, not a claim that any GGUF model is drop-in compatible.

Exact measurements remain visible without a model. Model jobs are bounded and
sampled; probes and changing conditions take priority over routine responses.
The first two free-form experiments failed: one output was structurally rejected,
and a later output cited the wrong event and invented an unsupported timing detail.
Free-form model narration is therefore **not enabled** in this version.

The active model only ranks two eligible predefined conditional explanations.
Code supplies their wording, relevant evidence IDs and read-only next observations.
Summary and facts are exact deterministic projections. Provenance explicitly says
`ranked_catalogue_selection`; these are not original model-written explanations.
Malformed selections fall back to the measurement summary. Correctly linked
candidates do not establish that the model's ranking is useful or causes are true.

Eido remains an evaluation candidate. Its latest inspected native 4B smoke
completed 3/6 episodes, with one unsupported typed claim, no fresh acceptance
cases and no scored free-text explanation quality. Those results do not justify
promoting it as a reliable evidence interpreter.

## Evidence and monitored boundary

- These are the observer's own HTTP checks, not visitor counts or captured flows.
- Latency measures client elapsed time through this runtime's outbound proxy. It
  is not origin processing time, direct-network latency, or browser page load.
- A returned HTTP 200 proves a response was received, not that login, rendering,
  checkout, internal services, or security controls work.
- Cloudflare visitor traffic, request logs, origin logs, internal networks and
  end-to-end TLS certificate inspection are **not connected**.
- Requested cadence, actual timestamps, source freshness, unavailable responses,
  missing coverage, queue skips and retention are visible.
- Request start and completion are distinct. Upstream HTTP Date is retained for
  inspection; freshness uses the collector clock, not the upstream clock.
- Response-control header presence is only an observation. Absence is not proof
  of exploitability. Older evidence may explicitly have no captured controls.

SQLite retains the latest 5,000 measurements with an append hash chain and
separate interpretation checksums. Startup verifies the chain. Checksums detect
accidental changes; they are not authenticated, externally anchored or tamper-proof.
Model interpretation may arrive after the immutable measurement. The browser can
retrieve a retained event by evidence ID after it leaves the snapshot window.

No response bodies, cookies, authorization headers, client IPs, query strings or
private logs are collected. Network failures retain a bounded error code rather
than a raw exception that might expose proxy configuration. Targets are fixed;
redirects cannot leave the HTTPS host. Local POSTs require matching Host/Origin.

## Brain reuse and prediction experiment

The adapter imports the unchanged NumPy `SentinelV3` from
`eidos/repo/src/eidos_brain/sentinel/sentinel_v3.py`. Each route has an independent
provisional baseline over `log1p(latency_ms)`. Missing and nonfinite values are
excluded, never converted into zero. Warmup is shown as calibration, never green
security clearance. The research fixture's fixed 47-frame periodicity probe is
disabled in the adapter. Reservoir geometry changes are not threat severity.

The full `EIDOS_BRAIN_UNIFIED_v0_4.7.02.py` and Sentinel Lab were inspected and
preserved. Their Torch batch engine is not claimed to run in this new live process.
Existing monolithic false-positive evidence and the draft guided Lab are not
requalified by this product experiment.

Sentinel's current-frame reconstruction residual is not a prediction. A separate
EWMA experiment issues a frozen next-observation latency estimate, then joins it
to a later real measurement and compares its error with persistence (the previous
latency). No model narrative can create a forecast outcome. Gaps are unscored.
Twenty scored outcomes unlock only a **provisional observed-window comparison**;
the application never claims established predictive skill or predicts attacks.
A process restart begins a new Brain epoch; retained observations are not replayed
to manufacture retrospectively issued live forecasts.

## Offline replay

While the local service is running:

```sh
python export_replay.py --output /absolute/path/Sentinel_Observatory_Replay.html
```

Or supply `--snapshot /absolute/path/capture.json`. The exported single HTML file
embeds the actual recorded measurements, disables probing and all live fetching,
and displays **RECORDED**. Its replay slider reveals only the selected historical
window. Replay is a review artifact, not evidence of a currently active collector.

## Integration boundary

This app lives alongside `apps/sentinel-lab`; no existing Lab routes, Works
accounts, checkout, databases, deployment configuration or Brain core files change.
There is no automatic production release or new persistent access. A future
Cloudflare connector should use explicitly approved, scoped read-only access and
retain dataset-specific sampling/coverage metadata. Private hosting, visitor-log
ingestion, operational alert calibration and longitudinal forecast validation
remain separate work.
