import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { ARTIFACT_VERIFIER } from "../lib/experiments/artifact-verifier";

test("immutable verifier checks bytes, missing and forbidden paths without opening raw inputs", () => {
  const root = mkdtempSync(join(tmpdir(), "eidos-verifier-"));
  const good = Buffer.from("frozen predictions\n");
  const digest = { bytes: good.length, sha256: createHash("sha256").update(good).digest("hex") };
  try {
    mkdirSync(join(root, "engine_artifacts"));
    writeFileSync(join(root, "engine_trace.jsonl"), good);
    writeFileSync(join(root, "engine_artifacts", "receipt.json"), good);
    const artifacts: Record<string, unknown> = { "engine_trace.jsonl": digest, "engine_artifacts/receipt.json": digest };
    const run = () => {
      writeFileSync(join(root, "run_manifest.json"), JSON.stringify({ artifacts }));
      const result = spawnSync("python", ["-c", ARTIFACT_VERIFIER, root], { encoding: "utf8", windowsHide: true });
      assert.equal(result.status, 0, result.stderr);
      return JSON.parse(result.stdout);
    };
    assert.equal(run().allMatched, true);
    artifacts["engine_trace.jsonl"] = { ...digest, sha256: "0".repeat(64) };
    artifacts["metrics.json"] = digest;
    artifacts["../secret"] = digest;
    artifacts["engine_artifacts/../../secret"] = digest;
    artifacts["labels_sealed.json"] = digest;
    const receipt = run();
    assert.equal(receipt.allMatched, false);
    assert.equal(receipt.matchedCount, 1);
    assert.equal(receipt.files.filter((file: { error?: string }) => file.error?.includes("allowlist")).length, 3);
  } finally {
    assert.equal(dirname(resolve(root)), resolve(tmpdir()));
    rmSync(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
