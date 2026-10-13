# Sentinel: connect a target and watch

This upgrade adds continuous, foreground observations from a public URL. A
companion uses the same interface for URLs and public IP addresses. Enter a
target, press **Connect**, and inspect the actual results as they arrive.

The hosted app is private:
https://sentinel-evidence-observatory.bmparent.chatgpt.site

| Connection | Runs from | Supported targets |
|---|---|---|
| Hosted | Cloudflare Worker, behind the existing owner-only Sites access policy | Public HTTP/S DNS names, ports 80/443 |
| Local companion | The computer running the Python command below | Public HTTP/S DNS names and public IPv4/IPv6, ports 80/443 |

Cloudflare Workers cannot directly fetch an IP-address URL. The hosted app
explains that limitation instead of recording a fictional IP measurement.
Private networks, localhost and reserved addresses are excluded in both modes.
An internal-network connector requires a separately approved monitoring scope.

## Run the URL/IP companion

From the repository root, with Python 3.11 or newer:

```sh
python apps/sentinel-observatory/live/local_server.py
```

Or double-click `Start_Sentinel_Windows.bat` in the companion folder to start
the service and open your browser. On macOS/Linux, run
`sh Start_Sentinel_Mac_Linux.sh`. Both require Python 3.11 or newer.

Open http://127.0.0.1:8766. No package installation, model download, API key or
account creation is required. Enter a URL or public IP that you own or are
authorized to monitor. Bare targets default to HTTPS; include `http://` for an
HTTP service. A direct HTTPS IP needs a certificate valid for that IP; certificate
verification is never bypassed. Ctrl-C stops the companion.

The earlier fixed-route Python app and optional local Qwen interpreter remain
available alongside this version; their source and evidence are preserved.

## How live observation works

- Up to four targets; one bounded HEAD request per target, then at least fifteen
  seconds before another. This is a stream of sampled observations, not packets.
- Pause/resume and remove controls cancel the browser request and stop future
  checks. A request already sent to the target may finish at the collector.
- Hidden tabs pause checks. Closing the tab stops this foreground session.
  Reloaded history is restored paused; it never silently restarts monitoring.
- Up to 500 actual observations remain in this tab's session storage when
  available. Export JSON before closing the tab if you need a portable record.
- The old seven-minute recorded capture remains under `/recorded` on the hosted
  app. It is separate from the live connection and never seeds live baselines.

Each event exposes its source, start/completion time, actual response or error,
elapsed time, selected response controls and missing coverage. An HTTP 200 is
not a working-login or security verdict. A timeout means the observer did not
receive a usable response; it does not establish a site outage. Redirects are
recorded but never followed, so the selected endpoint remains the boundary.

## Brain and interpretation

`brain.mjs` is an explicitly identified JavaScript port of the existing NumPy
Sentinel V3, configured for one log-latency feature and a 32-state reservoir.
Its generated initial weights and source hashes come from the Python engine.
`brain-parity.json` records numerical parity checks against Python, including the
earlier actual measurements and separately labeled test fixtures. It is a port,
not a claim that Python or the full unified Torch engine runs in the browser.

Every active connection gets a fresh epoch. Missing/incomplete measurements are
excluded, not padded. Calibration and changes remain experimental. A frozen
EWMA latency prediction is compared with persistence on a later observation;
browser request and forecast-issue timestamps use the same clock. Historical
records cannot produce retrospectively issued forecasts. Short samples and
correlated network timings do not establish predictive skill.

The live view uses deterministic evidence notes, conditional explanations and
suggested next observations. **No language model runs in this hosted upgrade.**
The earlier Qwen adapter remains available in the original local application;
it is not silently represented as connected here. No paid AI endpoint is called.

## Boundaries and privacy

Targets cannot contain credentials, query strings or fragments. Only HTTP/S on
ports 80/443 is accepted. No response body, cookie, auth header, visitor address,
packet content or private log is collected or forwarded. No scanning, port
enumeration, network-protection change or defensive action occurs.

The local companion validates all DNS answers and connects to an approved
address while preserving the original hostname for Host and TLS SNI. The hosted
collector checks public DNS before fetching but **cannot pin the later Worker
DNS resolution**. It does not claim complete DNS-rebinding protection. Keep it
owner-private and without privileged origins, VPC bindings or internal services.
Origin checks supplement the platform access policy; Origin is not identity.

The hosted fifteen-second limit is per Worker isolate and canonical target,
not a durable globally coordinated quota. Browser concurrency is bounded and
the service is owner-private. Broader sharing requires stronger authorization,
abuse controls and deliberately reviewed data exposure.

## Validation and publishing

```sh
node --test apps/sentinel-observatory/live/collector.test.mjs apps/sentinel-observatory/live/brain.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_local_server.py' -v
node apps/sentinel-observatory/live/verify_brain_parity.mjs
```

The build script combines the collector and exact web assets into the supported
Sites Worker ESM entry. It does not fetch a target, generate evidence or embed
credentials. Pass the preserved recorded HTML explicitly:

```sh
python apps/sentinel-observatory/live/build_worker.py \
  --output /absolute/site/worker/index.js \
  --recorded /absolute/site/recorded.html
```

Use the existing Site identity and owner-only access. This app needs no database,
new monitoring credential, private-network binding or scheduled service. Browser
rendering and real analyst usability require their own verification; DOM and
API checks alone do not establish visual quality or operational protection.
