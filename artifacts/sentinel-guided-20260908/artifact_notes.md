# Artifact handling notes

All raw synthetic evaluation files remain in period-raw-artifacts.zip with individual checksums. The main Drive ZIP is an immutable pre-upload-status snapshot. Latest journal and archive-status sidecars are uploaded beside it; its embedded manifest applies to its own contents.

Earlier failed attempts remain: missing local pyarrow, document locator mismatch, old dual-stack fetcher, stale dev-server modules, duplicate synthetic monitor headings, cp1252 console output, preview protection, external Sandbox billing and mounted Drive storage error. Zero-length collector-validation.log means the official successful validator emitted no output; collector-validation.json records exit status. Zero-length allocation-provider-response.json reflects a CLI 404 printed to stderr; it is not a JSON success receipt. provider-recovery.json records cancellation, while preview-budget/receipt.json and budget-admission-receipt.json record the authoritative HTTP 402 rejection and released admission.

Automatic approval review rejected removal of this task's generated Next.js cache directories with the reason "blocked by policy". No alternative deletion method was attempted. Direct authenticated Drive upload completed the archive instead.

Git contains the compact review subset: reports, manifests, test logs, metrics, receipts and screenshots. Large generated fixtures, vector caches, raw ZIPs and numerical arrays remain repo-local and in the Drive ZIP. The original dirty repository was not used for implementation. No model/core legacy file or sealed proof archive was modified. The task's temporary official Collector executable was removed after its successful forwarding test; validation and download checksums remain.
