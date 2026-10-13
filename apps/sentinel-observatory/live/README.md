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

Read-only provider checks against the current Sentinel Lab scope returned unavailable security configuration. No provider firewall-event feed is presently attached. The hosted panel explicitly reports this missing coverage.

## Qwen and Eido

Select **Enable local Qwen** to download Qwen3-0.6B onto this browser's device. Inference uses a dedicated worker and WebGPU with float16 support, through pinned Transformers.js3.7.1 and a pinned ONNX model revision. The first download is hundreds of MB and requires a compatible current Chrome/Edge device. Unsupported hardware/download failures display unavailable; collection continues.

The model ranks two eligible predefined explanations from bounded numeric evidence. Code supplies facts, conditional wording and retained evidence references. Sensor alerts and network flags can be ranked too. A rejected/malformed output returns deterministic evidence notes with a rejection receipt. Each accepted result reports the actual model, revision, inference timing, input IDs and catalogue provenance. There is no model-written security verdict or autonomous defensive action, and no paid model API is used.

The optional local CPU interpreter uses the existing tested Qwen GGUF adapter. Configure `OBSERVATORY_MODEL_PATH` to a compatible Qwen3 GGUF after installing the optional llama-cpp-python runtime. The local `/api/analyze` route accepts retained observation IDs, not arbitrary client-created evidence, and bounds/caches model work. Eido remains an evaluation candidate; this app does not represent an unqualified Eido checkpoint as loaded.

## Local public URL/IP companion

From the repository root with Python3.11+:

```sh
python apps/sentinel-observatory/live/local_server.py --open-browser
```

Or use the provided Windows/macOS/Linux launcher. Open http://127.0.0.1:8766. This observer validates every DNS answer and pins one numeric address while retaining original Host/TLS identity. Bare addresses default to HTTPS. HTTPS certificates must be valid for the supplied hostname/IP; no insecure retry occurs. Ctrl-C stops the service.

## Validation and build

```sh
node --test apps/sentinel-observatory/live/collector.test.mjs apps/sentinel-observatory/live/brain.test.mjs apps/sentinel-observatory/live/browser-interpreter.test.mjs apps/sentinel-observatory/live/security-events.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v
node apps/sentinel-observatory/live/verify_brain_parity.mjs
python apps/sentinel-observatory/live/build_worker.py --output /absolute/site/worker/index.js --recorded /absolute/site/recorded.html
```

Builds preserve the prior capture under `/recorded` and package exact inspected source. They do not generate telemetry. Keep the existing owner-only Sites access and protected Vercel collector; no sharing change is needed. Unit/DOM checks, real hosted API observations, actual model inference and hardware-specific rendered-browser testing are separate evidence levels.
