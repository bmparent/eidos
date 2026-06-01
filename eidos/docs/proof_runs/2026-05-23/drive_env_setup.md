# Google Drive Proof Mirror Setup -- 2026-05-23

## What was attempted

Codex checked whether this Windows machine had a real Google Drive Desktop filesystem mount for proof artifact mirroring.

## Findings

- Google Drive connector access is authenticated as `1brent.bm@gmail.com`.
- No Drive Desktop mount was present at `G:\My Drive`.
- No DriveFS config folder was present at `%LOCALAPPDATA%\Google\DriveFS`.
- No Google Drive Desktop process was running.
- `C:\Users\bmpar\Google Drive` exists, but it is a normal local folder, not a verified Drive Desktop mount.

## Installation attempt

Codex attempted:

```powershell
winget install --id Google.GoogleDrive --silent --accept-package-agreements --accept-source-agreements
```

The installer download and hash verification succeeded, but installation exited with:

```text
0x800704c7 : The operation was canceled by the user.
```

## Helper added

Use this helper after Google Drive Desktop is installed and signed in:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/configure_proof_drive_env.ps1 -PersistUser
```

The helper only persists `EIDOS_PROOF_DRIVE_DIR` after it finds a verified writable Drive Desktop-style root, normally `G:\My Drive`.

For a non-mutating check:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/configure_proof_drive_env.ps1 -CheckOnly
```

## Current status

The env var was not persisted because no verified Google Drive filesystem mount exists yet.

## Next validation

After Drive Desktop is installed and signed in, rerun:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/configure_proof_drive_env.ps1 -PersistUser
python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 1200 --out artifacts/proof_false_positive_control_2026_05
```

The regenerated `drive_manifest.json` should show `drive_copy_success: true` and a `drive_root` pointing at the verified Drive mount.
