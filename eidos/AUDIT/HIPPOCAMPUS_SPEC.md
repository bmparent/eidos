# Hippocampus (HDC/VSA) Spec

## Implementation
- Class: `Hippocampus` in `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py`.
- Encodes context `h_r` from reservoir state and content `h_x` from frame.
- Binding: element-wise sign multiplication `m_t = h_r ⊙ h_x`.
- Banks: per-regime or global depending on config.

## Recall / Similarity
- `recall_similarity()` compares bound vectors to memory bank.
- Similarity returned in `[-1, 1]` with familiarity score `chi`.

## Write Policy
- Writes gated by surprise, novelty threshold, and rate limiting.
- Optional write-on-green logic if enabled.

## Failure Modes
- If similarity becomes NaN, writes are suppressed.
- Empty bank shortcuts allow initial writes.
