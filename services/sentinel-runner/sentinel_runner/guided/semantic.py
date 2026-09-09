"""Pinned sentence embeddings, owner-contained cache, and exact passage retrieval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Reviewed immutable model revision; no remote model code is permitted.
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
ENCODER = f"{MODEL}@{REVISION}:full-token-mean.v1"


def encode(texts: list[str], cache_dir: Path) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    cache_dir.mkdir(parents=True, exist_ok=True)
    output: list[list[float] | None] = [None] * len(texts)
    missing = []
    for i, text in enumerate(texts):
        key = hashlib.sha256((ENCODER + "\0" + text).encode()).hexdigest()
        path = cache_dir / f"{key}.json"
        if path.is_file():
            vector = json.loads(path.read_text())
            if len(vector) == 384 and np.isfinite(vector).all():
                output[i] = vector
                continue
        missing.append((i, path))
    if missing:
        model = SentenceTransformer(MODEL, revision=REVISION, device="cpu", trust_remote_code=False)
        # Long words can exceed the encoder window even in a 100-word passage.
        # Embed every token window, then pool; never silently truncate a suffix.
        pieces, ranges = [], []
        window = int(model.max_seq_length) - 2
        for i, _ in missing:
            tokens = model.tokenizer.encode(texts[i], add_special_tokens=False)
            start = len(pieces)
            pieces.extend(model.tokenizer.decode(tokens[j:j + window]) for j in range(0, len(tokens), window))
            ranges.append((start, len(pieces)))
        encoded = model.encode(pieces, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        values = []
        for start, end in ranges:
            value = encoded[start:end].mean(axis=0)
            values.append(value / max(float(np.linalg.norm(value)), 1e-12))
        for (i, path), vector in zip(missing, values):
            output[i] = vector.astype(float).tolist()
            path.write_text(json.dumps(output[i]), encoding="utf-8")
    return output


def analyze(dataset: dict, artifact_dir: Path) -> dict:
    chunks = dataset["passages"]
    vectors = np.asarray(encode([c["text"] for c in chunks], artifact_dir / "embedding-cache"))
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1)
    neighbors = []
    for i, chunk in enumerate(chunks):
        candidates = np.argsort(-similarities[i])[:3] if len(chunks) > 1 else []
        neighbors.append({"passageId": chunk["id"], "neighbors": [
            {"passageId": chunks[int(j)]["id"], "similarity": float(similarities[i, j])} for j in candidates if j != i]})
    return {"schema": "eidos.semantic.v1", "method": "Sentence Transformer semantic embedding and cosine passage retrieval",
            "sourceBadge": "Your documents · pinned semantic encoder", "encoder": ENCODER,
            "vectors": vectors.tolist(), "passageIds": [c["id"] for c in chunks], "neighbors": neighbors,
            "findings": [{"schema": "eidos.finding.v1", "id": "document-index", "what": f"{len(chunks)} source passages indexed for retrieval.",
                          "whyItMatters": "Related passages can be inspected with their original source and page references.",
                          "basis": "actual extracted passages and pinned semantic vectors", "score": None, "entity": dataset["source"],
                          "members": [{"passageId": c["id"]} for c in chunks], "detector": ENCODER,
                          "uncertainty": ["Similarity is association, not factual verification or causal evidence."],
                          "nextAction": "Ask a bounded question and inspect the cited passages.", "grouping": "Single document source, original passage boundaries."}],
            "metrics": {"passages": len(chunks), "dimensions": 384, "llmCalls": 0, "llmCostUSD": 0, "precision": None, "recall": None},
            "forecasts": [], "observations": [], "gatesAdvanced": 0,
            "limitations": ["Source content is untrusted data. Retrieval does not establish its truth.",
                            "Image-only PDFs and OCR are outside this profile. No temporal Eidos evidence is claimed for documents."]}


def retrieve(dataset: dict, result: dict, question: str, artifact_dir: Path) -> dict:
    if not 1 <= len(question.strip()) <= 500:
        raise ValueError("QUESTION_LIMIT: enter 1–500 characters.")
    if result.get("encoder") != ENCODER:
        raise ValueError("ENCODER_VERSION_MISMATCH: rebuild the document index.")
    query = np.asarray(encode([question], artifact_dir / "embedding-cache")[0])
    vectors = np.asarray(result["vectors"])
    if vectors.shape != (len(dataset["passages"]), 384):
        raise ValueError("INDEX_INTEGRITY_FAILURE")
    scores = vectors @ query
    indices = np.argsort(-scores)[:5]
    citations = [{**dataset["passages"][int(i)], "similarity": float(scores[i])} for i in indices]
    return {"schema": "eidos.answer.v1", "method": "semantic retrieval; deterministic extractive answer", "question": question,
            "answer": "These are the closest source passages. They are quotations from the imported source, not verified conclusions.",
            "citations": citations, "encoder": ENCODER, "llmCalls": 0,
            "limitations": ["The source may not answer the question. Similarity is not confidence in a factual claim."]}
