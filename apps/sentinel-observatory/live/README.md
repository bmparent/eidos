# Sentinel Observatory: connect, watch, inspect

Private live app: https://sentinel-evidence-observatory.bmparent.chatgpt.site

Enter a public URL or IP, select **Connect**, and watch evidence arrive. **Test public IP** selects Cloudflare's public 1.1.1.1 HTTPS endpoint. Each visual mark opens the original observation, response, timing, collection boundary and Sentinel analysis. Pause, remove and export controls preserve the collected evidence.

## Live endpoint observations

The hosted app collects public HTTP/S DNS-name endpoints in its Cloudflare Worker. Public literal IPs use a separately authenticated Node collector in the existing Sentinel Lab Vercel preview. The Node collector connects to the validated numeric IP, performs one HEAD request, verifies TLS against that IP, and never follows redirects or reads response bodies. HTTP uses port80 and HTTPS443. Private, reserved, local and special-use addresses are rejected.

Relay configuration is server-only: `OBSERVATORY_COLLECTOR_URL`, `OBSERVATORY_COLLECTOR_SECRET` and, where Vercel Deployment Protection applies, `OBSERVATORY_COLLECTOR_BYPASS_TOKEN`. Keep all values outside source and browser assets. Without the relay, arbitrary hosted IP monitoring reports unavailable. The explicit 1.1.1.1 named-service fallback is labeled DNS-routed and does not claim a pinned-IP connection.

There are at most four targets. Checks run at least15 seconds after the previous result, only while this browser session is connected and visible. Hidden tabs pause; reopening restores retained evidence paused. There is no unattended/background monitoring. The latest500 endpoint records remain in browser session storage where available; export JSON for a portable copy.

These are the observer's checks. They do not reveal a target's visitor traffic, internal network or attacks. A returned HTTP200 is a received response, not proof of functional login or security. A timeout preserves unknown target availability. The hosted DNS-name path checks all DNS answers but cannot pin Worker DNS resolution; the literal-IP relay pins its destination without DNS.

## Network and security sensors

Open **Network & security** to import an existing Suricata EVE JSON/JSONL or Zeek JSON log. No example incident is automatically inserted. The panel shows real sensor detections, observed network connections, traffic relationships, source timestamps, retained evidence and source coverage. Imported logs are historical evidence. A rule match or Zeek notice is a sensor detection, not a confirmed incident.

The normalizer accepts Suricata alert/flow/DNS/TLS and Zeek connection/notice/DNS/TLS metadata. It excludes payloads, HTTP headers, cookies, certificates and arbitrary notice messages. Internal addresses and DNS names may still appear locally; imported evidence remains in the current browser memory until export. The parser bounds bytes, line sizes, records and timestamps, records malformed/unsupported rows, and deduplicates evidence.

Unusual traffic uses an explicitly identified port of unchanged Sentinel V3 on per-source log-transformed connection bytes. Warmup and provisional numerical changes remain experimental. A separate high-destination-count indicator is an explicit fanout heuristic, not proof of scanning or an attack. Missing coverage is visible, and disconnected sources do not imply zero events.

For a live existing sensor, set `OBSERVATORY_SENSOR_LOG` to the absolute path of its Suricata EVE or Zeek JSON file before starting the local companion. The companion reads a bounded regular-file tail every requested refresh, strips fields, and exposes `/api/security-events` only on loopback. It does not start Suricata/Zeek, sniff packets or select files from HTTP input. The browser can start/stop the sensor feed; stale/error states preserve previous evidence without claiming fresh coverage.

## Attached application-edge sensor

The hosted Observatory now has a live application-edge sensor backed by its owner-private event store. It records actual `/api/observe` requests, their response outcomes, requested outbound HEAD receipts and known controls that deny or rate-limit a request. DNS-name and IP checks are linked to original measurement IDs. Literal-IP records include public peer identity, actual socket byte counters and TLS status when supplied by the authenticated Node collector.

The network panel automatically reads this attached source when opened. Reading the feed, static assets and capability requests do not generate traffic records. Pause controls and source-event age remain distinct from a successful feed refresh. Imported Suricata/Zeek and local log sensors remain available alongside this source.

Scope is this Observatory's requests and outbound checks. It does not capture LAN packets, remote target visitor traffic or provider firewall incidents. A recorded deny/rate-limit event proves the stated control ran; it does not prove an attack. Deliberate verification requests are labeled functional checks. Raw bodies, credentials, client addresses, user agents and target paths are excluded.

The store retains at most1000 records and exposes a seven-day window. Expired rows are purged on the next sensor read/write; no unattended cleanup is promised. Storage or query failures show unavailable. The schema is an additive migration, with no runtime DDL or inserted sample incidents.

Read-only provider checks against the current Sentinel Lab scope returned unavailable security configuration. No provider firewall-event feed is presently attached. The hosted panel explicitly reports this missing coverage.

## GPT-6.1 Sol and Eido

The hosted interpreter uses the exact OpenAI model `gpt-6.1-sol` through a server-side Responses API call. Select retained evidence, then explicitly request interpretation. It does not download a browser model or require Hugging Face access. `OPENAI_API_KEY` belongs in the existing Site's secret runtime environment; it is never supplied to browser code. A missing key displays **not configured** and preserves deterministic measured-evidence notes. A configuration check alone does not prove model access or successful inference.

The model ranks two eligible predefined explanations from bounded numeric and enum metadata in authoritative server-retained native records. Imported logs and client-supplied measurements cannot trigger paid inference. Endpoints, IP addresses, raw URLs, request bodies, headers and arbitrary log text stay outside model context. Code supplies facts, conditional wording and evidence references. Accepted output includes provider/model identity, timing, usage, input IDs and prompt/output hashes. Missing, malformed or unverifiable output displays evidence fallback; an attempted call with no verified receipt is not represented as a proven model run.

Paid calls run manually, with low reasoning effort and at most1024 output tokens. Durable per-evidence reservations cache results and prevent repeat calls. A60-second cooldown and UTC daily limits of20 reservations or$0.50 in conservative reservations bound usage; at$0.05 reserved per call, the budget currently permits at most10 calls. Reservations are conservative application controls, not an invoice or provider billing cap. Failed attempts remain reserved and are not automatically retried. Model-job records expire after eight days on the next valid retained-evidence interpretation request; expired seven-day source records cannot trigger inference again. There is no model-written security verdict or autonomous defensive action.

The optional local CPU interpreter uses the existing tested Qwen GGUF adapter. Configure `OBSERVATORY_MODEL_PATH` to a compatible Qwen3 GGUF after installing the optional llama-cpp-python runtime. The local `/api/analyze` route accepts retained observation IDs, not arbitrary client-created evidence, and bounds/caches model work. Historical browser Qwen source and receipts remain available as prior implementation evidence; hosted interpretation does not invoke them. Eido remains an evaluation candidate; this app does not represent an unqualified Eido checkpoint as loaded.

Credential provisioning and the remaining actual-provider/browser checks are documented in [CODEX-HANDOFF-SOL.md](./CODEX-HANDOFF-SOL.md).

## Local public URL/IP companion

From the repository root with Python3.11+:

```sh
python apps/sentinel-observatory/live/local_server.py --open-browser
```

Or use the provided Windows/macOS/Linux launcher. Open http://127.0.0.1:8766. This observer validates every DNS answer and pins one numeric address while retaining original Host/TLS identity. Bare addresses default to HTTPS. HTTPS certificates must be valid for the supplied hostname/IP; no insecure retry occurs. Ctrl-C stops the service.

## Validation and build

```sh
node --test apps/sentinel-observatory/live/collector.test.mjs apps/sentinel-observatory/live/brain.test.mjs apps/sentinel-observatory/live/browser-interpreter.test.mjs apps/sentinel-observatory/live/security-events.test.mjs apps/sentinel-observatory/live/hosted-sensor.test.mjs apps/sentinel-observatory/live/sol-interpreter.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v
node apps/sentinel-observatory/live/verify_brain_parity.mjs
python apps/sentinel-observatory/live/build_worker.py --output /absolute/site/worker/index.js --recorded /absolute/site/recorded.html
```

Builds preserve the prior capture under `/recorded` and package exact inspected source. They do not generate telemetry. Keep the existing owner-only Sites access and protected Vercel collector; no sharing change is needed. Unit/DOM checks, real hosted API observations, actual model inference and hardware-specific rendered-browser testing are separate evidence levels.
