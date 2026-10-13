# Sentinel Observatory live IP, interpreter and sensor acceptance

Requested: finish live URL/IP monitoring, integrate lightweight interpretation, add networking/security evidence and intuitive visual inspection. Observatory remains owner-private. Existing Brain, Eido research and Sentinel Lab behavior are preserved.

## Candidate and observed services

- Private live app: https://sentinel-evidence-observatory.bmparent.chatgpt.site
- Source branch: `codex/sentinel-observatory-20261013`, PR73 in `bmparent/eidos`.
- Exact pinned-IP Node backend: `c40fb741a92d0dbe4a812692c82e5845a5ccb092`, Vercel deployment `dpl_9wYALbR1LGgughoSLBHV7CgqtM7R`, protected immutable preview `eidos-sentinel-lzzhgixpu-1brentbm-1876s-projects.vercel.app`.
- Initial integrated hosted acceptance: Sites source `7a5fb909b8f537e3302ef99828899e0f9dd6a51b`, deployment `appgdep_6ace912814c8819192bc0bf97839dfc0`, environment revision3.
- Final reviewed UI: Sites source `3624fffd27312385c922cbbcb7684deffb216932`, deployment `appgdep_6ace92888b2c819186ff97e99ad21d4a`, succeeded2026-10-13T20:20:30Z, environment revision3. Native receipt is `evidence/2026-10-13/hosted-final-deployment.json`.

## Evidence ledger

| Requirement | Observation | State |
|---|---|---|
| Actual public-IP monitoring | Hosted Sites→authenticated Vercel collector→literal `https://1.1.1.1/`: three HEAD301 results at50.145,32.777,25.127 ms; IP pinned, no error or redirect follow | Hosted/provider verified |
| Live URL regression | Hosted `https://eidos-works.com/` returned actual HEAD200 | Hosted verified |
| Public-only authenticated boundary | Unauthenticated Sites API403; loopback target400; Node fail-closed auth, public ranges, IP certificate identity, no redirects/bodies, deadlines and capacity tests | Pass |
| Genuine model inference | Qwen3-0.6B Q8_0 actually ran on the fresh retained IP301 event and accepted `edge,path`; exact input and model/prompt/output hashes retained | Actual local CPU inference verified |
| UI to collector to model | Exact live UI in DOM→local actual HEAD transport-error receipt→actual Qwen CPU inference→original evidence-linked result→pause retained record | DOM/API/model verified; not a pixel/browser run |
| Network/security tools | Suricata EVE and Zeek JSON imports + explicitly configured read-only local log tail; source timestamps, detection records, topology, volume changes and fanout evidence | Implemented, parser/tail/DOM verified |
| Sensor coverage | No actual sensor file or packet feed is attached to hosted service. Current provider security-config/attack-status checks returned unavailable configuration, not zero attacks | External data-source gap |
| Interpretation integrity | Fixed predefined candidate wording, exact facts and citations; raw URLs/addresses/signatures/payloads excluded from model context; rejected output identified as fallback | Pass |
| Browser Qwen | Pinned ONNX revision and Transformers.js3.7.1, dedicated worker, explicit opt-in, progress/hardware checks | Implemented/controller tested; WebGPU download/inference unverified on user hardware |
| Eido | No trusted Eido checkpoint loaded; correctly marked evaluation candidate | Not promoted |
| Unattended monitoring | Foreground browser checks; hidden tabs pause, reload restores paused; sensor receipt and source-event age remain distinct | Explicit product boundary |

Original actual evidence is under `evidence/2026-10-13/`. These files contain no provider credentials. Test-only records are used only by unit/DOM tests, never seeded into live telemetry. The first backend build failed strict typing; the corrected exact backend passed and deployed READY. Failed test/build observations are not counted as success.

## Validation

-45 changed-path Node tests:28 hosted collector,7 model-controller,10 sensor adapter.
-28 Python local collector/model/sensor tests.
-10 Node backend boundary/transport tests plus isolated strict TypeScript check.
-17 unchanged portable Brain tests; canonical Python numerical parity across954 frames and41,022 fields.
-Parent UI and network-panel DOM regressions passed. Independent review issues fixed:2000 aggregate retention cap, source freshness/paused/error state, Qwen hypotheses/provenance rendering, and accurate fixed-threshold fanout wording.
-Source syntax, whitespace/diff and Worker artifact export checks passed.

These checks do not establish incident-detection precision, model ranking usefulness, longitudinal forecast skill, physical-device rendering or browser-GPU inference. An imported sensor alert is a recorded detection, not an independently confirmed incident. Current missing sensor coverage remains visible.
