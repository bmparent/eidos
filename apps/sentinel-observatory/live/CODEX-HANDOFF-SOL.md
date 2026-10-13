# Sentinel Observatory: activate GPT-6.1 Sol and finish live acceptance

Continue the existing Observatory. Do not rebuild it or replace its evidence with demo data.

Repository: `bmparent/eidos`. Draft pull request: https://github.com/bmparent/eidos/pull/73. Branch: `codex/sentinel-observatory-20261013`.

Private app: https://sentinel-evidence-observatory.bmparent.chatgpt.site

Sites project: `appgprj_6ace6f7f4a5481918adb34a5cb366de3`. Preserve its current owner-only access.

## Remaining task

The user authorized using the OpenAI Developers plugin to provision an API key and chose **GPT-6.1 Sol** for interpretation. That plugin is unavailable in this ChatGPT Work session; the user requested a Codex handoff for credential setup. No OpenAI API key was generated here, and no live Sol inference is claimed.

Use the connected OpenAI Developers plugin's supported key-provisioning workflow in Codex. Read its instructions and the `openai-platform-api-key` skill when available. Keep the key out of messages, source, browser assets, shell arguments, temporary files and logs. Never reuse an unrelated product's key. If provisioning requires an account/project decision that cannot be inferred, resolve that decision before creating the key.

Store it as the **server runtime secret `OPENAI_API_KEY`** for this existing Sites project. Use the Sites environment-variable operation, with `is_secret: true`, and deploy a saved version to apply the changed environment revision. Preserve the existing `OBSERVATORY_COLLECTOR_*` secrets and `OBSERVATORY_DB` D1 binding. Do not put runtime secrets into `.openai/hosting.json`.

The implementation uses the Responses API at `https://api.openai.com/v1/responses`, exact model `gpt-6.1-sol`, low reasoning effort, structured output, no tools and no background inference. Model interpretation is a manual action. Requests accept only an authoritative retained evidence ID; imported/client-created records cannot create a paid inference request. Durable caching, concurrency exclusion and daily reservations bound work. Keep those limits in place for live acceptance.

## Live verification

1. Confirm `/api/capabilities` reports the hosted Sol interpreter configured. Configuration alone does not prove provider access.
2. Connect the existing public test `https://1.1.1.1/`, or use a recent retained native collector record. Preserve the actual first-hop HTTP status, redirect uncertainty, verified TLS, public peer identity and socket counters. Do not scan a random address.
3. Select that evidence and explicitly request interpretation. Verify a real accepted provider response, `model_ran: true`, `output_accepted: true`, exact `gpt-6.1-sol` provenance, retained input evidence IDs and conditional catalogue explanations. Facts must remain derived from the stored evidence.
4. Request the same evidence again. Verify the cached receipt and no second provider call. Check unknown/imported IDs, extra client fields, cross-origin requests and quotas reject safely without inference.
5. Verify a missing/invalid key or provider failure displays unavailable/fallback evidence, never a model result or a zero-security-events claim. Do not test failures by exposing or deleting credentials.
6. Inspect the actual rendered app on desktop and narrow mobile: connect/pause/resume, selected-evidence interpretation, sensor freshness, request/connection/control filters, evidence inspector, export and focus/keyboard controls. DOM tests in Work are not pixel or end-user browser acceptance.
7. Save sanitized provider/deployment receipts and an updated acceptance report under this app's `live/evidence/`. Record exact product source SHA, Sites source SHA, deployment ID and environment revision. Never include keys, bearer tokens, raw request content or model input endpoints in receipts. Update the existing draft PR; do not merge unrelated work.

## Source and packaging

Core source is `apps/sentinel-observatory/live/`. `sol-interpreter.mjs` owns the server call and bounded evidence interpretation. `hosted-sensor.mjs` owns authoritative retained native records. `web/app.js` and `web/network-panel.mjs` display measured evidence and interpretation provenance. `build_worker.py` packages inspected sources; it generates no traffic.

Additive SQL migrations are in `live/migrations/`, with immutable applied migration history. The Sites source checkout uses `drizzle/` and its journal. Use the Sites skill/source helper to open the existing source, copy the migration files and build the Worker, then push and publish that exact source. Keep the prior recorded capture at `/recorded`.

The literal-IP relay is already deployed in the existing protected Sentinel Lab Vercel preview. Do not weaken Deployment Protection or replace the relay. DNS-name Worker requests have preflight validation but cannot pin the subsequent DNS resolution; this remains a visible limitation.

## Evidence boundaries

The attached sensor records actual Observatory requests, outbound checks and enforced boundary/rate controls. Deliberate audit requests are marked functional checks. These records are not confirmed attacks or a packet capture of the monitored IP. Suricata/Zeek imports and a configured local existing-log source can add the network metadata they actually contain. Eido remains an evaluation candidate, not a loaded accepted checkpoint.

Historical Qwen tests and receipts remain historical. The hosted default now uses Sol; it must not download a Hugging Face model to interpret evidence. A compatible optional local CPU Qwen companion remains separate.

Official model reference: https://developers.openai.com/api/docs/models/gpt-6.1-sol

Structured-output reference: https://developers.openai.com/api/docs/guides/structured-outputs
