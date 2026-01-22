# Smoke Test Results

## Command
- `python tools/demo_smoke_test.py`

## Result
- **FAILED**: missing `torch` dependency.
- Error: `ImportError: torch is required to run the Eidos engine. Install with pip install torch.`

## Notes
- The smoke test script is in place but cannot complete in this environment without torch.
