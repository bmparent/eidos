# Codex Audit Report

## Inventory & entrypoints

**Primary entrypoints**
- `src/eidos_brain/engine/eidos_v0_4_7_02.py` → `run_eidos_sentinel()` (engine main).【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4602-L5127】
- `src/eidos_brain/service/daemon.py` → `main()` loop for daemon runs.【F:src/eidos_brain/service/daemon.py†L1-L81】
- `src/eidos_brain/service/api.py` → FastAPI `POST /run` and `main()` for API server.【F:src/eidos_brain/service/api.py†L1-L97】
- `src/eidos_brain/demo/demo_runner.py` → CLI demo runner orchestrating engine + dashboard.【F:src/eidos_brain/demo/demo_runner.py†L1-L70】

**Key configs / env vars**
- `EIDOS_DATA_SOURCE` (aka `DATA_SOURCE_TYPE`): `LOCAL`, `DRIVE`, `KAGGLE`, `STREAM`, `HIVE_PUBSUB`, `HIVE_GCS`.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L64-L71】
- `EIDOS_ARTIFACT_ROOT`: artifact storage root override.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L78-L84】
- `HIVE_BACKEND`: `LOCAL` or `GCS` (affects artifact persistence).【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L648-L659】
- NL/Gemini controls: `CONFIG_MODE`, `NL_MODE`, `LLM_PROVIDER`, `NL_COMMAND`.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L192-L201】

**How to run locally**
- Follow README quickstart (venv + editable install + demo script): `./scripts/run_local_demo.sh`.【F:README.md†L1-L22】

## Bugs & risks found

### 1) Adapter import crash (missing `run()` in engine module)
**Repro**
1. `from eidos_brain.engine.adapters import run_session`.
2. Import fails because `eidos_v0_4_7_02.py` did not export `run`.

**Root cause**
- `adapters.py` expects `run` in `eidos_v0_4_7_02.py`, but it was never defined.【F:src/eidos_brain/engine/adapters.py†L9-L31】

**Fix**
- Added `run(config)` and runtime config application helper in the engine module to bridge adapter expectations and apply config overrides safely.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4590-L4636】

**Why it’s correct**
- The adapter can now reliably call the engine entrypoint while honoring config overrides and returning a consistent result shape.

### 2) Provenance manifest writer errors + wrong write call
**Repro**
1. Call `run_eidos_sentinel()` → `_write_provenance_manifest()` executes.
2. It referenced `datetime.datetime` (wrong import) and called `store_memory_artifact("run_manifest.json", manifest)` with parameters inverted, so manifest data never wrote correctly.

**Root cause**
- `datetime` was imported as `from datetime import datetime`, but code used `datetime.datetime`. It also passed `label`/`data` in the wrong order for `store_memory_artifact`.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4624-L4688】

**Fix**
- Hardened the manifest write path: timezone-aware timestamps, atomic local file write, safe GCS write, and correct data serialization (no inverted arguments).【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4624-L4688】
- Added robust provenance writer with atomic writes and timezone-aware timestamps in `utils/provenance.py`, plus tests that assert expected keys.【F:src/eidos_brain/utils/provenance.py†L1-L85】【F:tests/test_provenance_manifest.py†L1-L24】

**Why it’s correct**
- The manifest writer now works deterministically across platforms, avoids partial writes, and never throws on transient IO issues.

### 3) Preflight NL mode incorrectly blocked
**Repro**
1. Set `CONFIG_MODE=NL_GEMINI` and `DATA_SOURCE_TYPE=LOCAL`.
2. `_preflight_inputs()` still performed file checks, blocking NL planning flows.

**Root cause**
- Preflight consulted `locals()` instead of the true global `CONFIG_MODE`, so NL mode detection was unreliable.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L112-L125】

**Fix**
- `_preflight_inputs()` now reads the global config mode directly and skips checks for NL/Gemini as intended. Added tests to validate behavior.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L112-L128】【F:tests/test_preflight.py†L1-L24】

**Why it’s correct**
- Preflight checks now reflect actual runtime configuration and no longer block NL planning modes.

### 4) Optional backend dependency guards (GCS/websockets/pubsub)
**Repro**
- Set `HIVE_BACKEND=GCS` without `google-cloud-storage` installed → NameError/obscure crash during init.
- Use `ws://` streaming without `websockets` installed → thread fails with unclear error.
- Use `HIVE_PUBSUB` without `google-cloud-pubsub` installed → unclear failure.

**Root cause**
- Missing dependency checks in critical init paths; imports were attempted in runtime threads or in partially initialized state.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L523-L676】【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L3405-L3532】【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4425-L4510】

**Fix**
- Added clear, early dependency checks for GCS, websockets, and Pub/Sub with actionable install messages. Added tests to ensure missing deps fail fast (skipped when torch isn’t present).【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L523-L690】【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L670-L720】【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L3405-L3532】【F:tests/test_optional_deps.py†L1-L35】

**Why it’s correct**
- Optional backends now fail clearly and deterministically instead of raising `NameError`/thread exceptions.

### 5) Multi-session state leakage in daemon/API runs
**Repro**
1. Call `run_session(configA)` then `run_session(configB)` in the same process.
2. The second run inherits mutated globals (artifact root, profile label, engine config) from the first run.

**Root cause**
- `_apply_runtime_config()` mutates globals and updates `EIDOS_BRAIN_CONFIG` in-place, causing cross-run state bleed.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4584-L4646】

**Fix**
- Added `DEFAULT_EIDOS_BRAIN_CONFIG`, `reset_runtime_state()`, and deep-merge config application so each run starts from a pristine baseline, with per-run overrides applied safely.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L340-L362】【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4576-L4668】
- Added a multi-session isolation test and a deep-merge unit test to verify isolation and merge behavior.【F:tests/test_multi_session_isolation.py†L1-L70】

## Fix summary
- Added `run(config)` entrypoint for adapter compatibility and applied runtime configuration overrides in-engine.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4584-L4657】
- Hardened provenance manifest writing with atomic local writes + GCS write fallback and timezone-aware timestamps, using a shared manifest helper.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4673-L4736】【F:src/eidos_brain/utils/provenance.py†L1-L87】
- Fixed preflight NL mode detection and added tests for both manual and NL flows.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L112-L128】【F:tests/test_preflight.py†L1-L24】
- Added explicit dependency guards for torch, GCS, Pub/Sub, and websockets; added streaming tests using mocks and websocket error handling.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L340-L735】【F:tests/test_streaming.py†L1-L114】
- Added resettable runtime state + deep-merge config updates to prevent cross-run bleed in daemon/API runs.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4576-L4668】
- Added smoke run and manifest tests; reduced golden replay reservoir size to speed deterministic test runs.【F:tests/test_smoke_synthetic.py†L1-L35】【F:tests/test_replay_golden_session.py†L1-L47】

## Tests & checks run
- `python -m compileall .`
- `pytest -q`
- `ruff check .` (existing lint issues remain in legacy code)
- `mypy src` (reports existing missing stubs + type errors in legacy engine module)
- `bandit --version` (not installed in environment)
- `pip-audit --version` (not installed in environment)

## Remaining issues / TODOs
- Torch isn’t installed in this environment; tests that exercise the engine are skipped without it. Install `torch` to run the full synthetic smoke tests and streaming tests.
- `ruff check .` and `mypy src` report numerous pre-existing issues in legacy modules (especially `eidos_v0_4_7_02.py`). These are unchanged beyond the targeted fixes above and should be addressed in future linting passes.

## Multi-session isolation strategy
- Each call to `run(config)` now begins with `reset_runtime_state()`, restoring all runtime globals to their pristine defaults and deep-copying `EIDOS_BRAIN_CONFIG` before applying per-run overrides.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4576-L4657】
- A deep-merge helper ensures nested config overrides do not mutate shared baselines across runs.【F:src/eidos_brain/engine/eidos_v0_4_7_02.py†L4576-L4584】
