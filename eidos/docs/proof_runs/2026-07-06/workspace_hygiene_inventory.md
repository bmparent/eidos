# Workspace Hygiene Inventory - 2026-07-06

This is a read-only inventory of workspace noise observed while executing the Month 1 final proof package. No files were deleted, moved, quarantined, or rewritten as part of this inventory.

## Scope

- Noisy original workspace inspected: `C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos`
- Clean execution worktree used for merge/package work: `C:\Users\bmpar\OneDrive\Documents\eidos-next-steps-execution-2026-07-06`
- Inventory command shape: `git status --short` plus filtering for `(1)`, `.pyc`, and `__pycache__`

## Findings

- Total original-workspace status lines: `251`
- Duplicate parenthesized `(1)` paths: `247`
- Modified pyc/cache paths: `3`
- The clean execution worktree did not carry the original workspace's duplicate `(1)` files into the merge/package work.

## Duplicate Distribution By First Path Segment

| Segment | Count |
| --- | ---: |
| `repo` | 92 |
| `..` | 57 |
| `docs` | 34 |
| `eidos-life-lab` | 25 |
| `tests` | 17 |
| `tools` | 8 |
| `sentinel` | 5 |
| `proof` | 3 |
| `scripts` | 2 |
| `benchmarks` | 2 |
| `.gitignore (1)` | 1 |
| `eidos_tensor_utils (1).py` | 1 |

## Pyc / Cache Churn Observed In Original Workspace

```text
M __pycache__/EIDOS_BRAIN_UNIFIED_v0_4.7.02.cpython-311.pyc
M repo/src/eidos_brain/__pycache__/__init__.cpython-311.pyc
M repo/src/eidos_brain/engine/__pycache__/eidos_v0_4_7_02.cpython-311.pyc
```

## Interpretation

The duplicate `(1)` files appear to be pre-existing OneDrive-style copies and generated artifact duplicates, not files created by the clean merge/package worktree. They should remain visible for auditability until a separate cleanup plan is explicitly approved.

## Recommended Follow-Up

Create a separate hygiene-only task that inventories exact duplicate pairs, classifies each as generated artifact, tracked-file duplicate, or unrelated experiment output, and then proposes a reversible archive plan. Do not delete these files as part of proof-package work.

## Proof Logic + Meaning

Goal reached: workspace noise was inventoried without modifying the noisy original checkout. Gate status is `partial` because this is an inventory, not cleanup.

Previous state: duplicate `(1)` files and pyc churn were visible in the original checkout and could confuse proof receipts if work continued there.

Technical logic utilized: read-only `git status --short` classification by filename pattern and first path segment.

Math / scoring logic:

```text
duplicate_ratio = duplicate_parenthesized_1_lines / total_status_lines
duplicate_ratio = 247 / 251
```

Philosophical meaning: auditability before cleanup. Proof work should preserve uncertainty before deciding what to remove.

Why this is better: the proof package now explains why a clean sibling worktree was used and preserves the noisy checkout state as evidence.

How this moves Eidos closer to the north-star goal: reproducibility improves when proof receipts are separated from local workspace noise.

Evidence: the inventory counts above came from the original workspace status output captured during this run.

Remaining uncertainty: the exact origin and duplicate-pair mapping for each `(1)` file remains unproven and should be handled in a separate hygiene task.
