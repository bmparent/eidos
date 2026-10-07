import { randomBytes, createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

const root = resolve(import.meta.dirname, "../../..");
const envPath = resolve(root, "apps/sentinel-lab/.env.local");
if (existsSync(envPath)) throw new Error(".env.local already exists; preserve it and configure guided variables explicitly.");
const privateDir = resolve(root, "artifacts/sentinel-guided-private");
mkdirSync(privateDir, { recursive: true });
const keys = Object.fromEntries(["alice", "bob"].map(user => [user, "eidos_test_" + randomBytes(32).toString("base64url")]));
const grants = Object.entries(keys).map(([id, key]) => ({ id: `qa-${id}`, sha256: createHash("sha256").update(key).digest("hex"), expiresAt: new Date(Date.now() + 24 * 3600000).toISOString() }));
writeFileSync(resolve(privateDir, "local-access.json"), JSON.stringify(keys), { mode: 0o600 });
const values = {
  EIDOS_GUIDED_LOCAL: "1", EIDOS_GUIDED_DATABASE_URL: `file:${resolve(root, "artifacts/guided-local.db").replaceAll("\\", "/")}`,
  EIDOS_GUIDED_SCOPE: "local:guided-v1", EIDOS_GUIDED_PYTHON: resolve(root, process.platform === "win32" ? ".venv-guided/Scripts/python.exe" : ".venv-guided/bin/python").replaceAll("\\", "/"),
  EIDOS_GUIDED_REPO_ROOT: root.replaceAll("\\", "/"), EIDOS_GUIDED_JOB_ROOT: resolve(root, "artifacts/guided-jobs").replaceAll("\\", "/"),
  EIDOS_SOURCE_COMMIT: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
  EIDOS_TEST_ACCESS_GRANTS: JSON.stringify(grants),
  EIDOS_GUIDED_CALLBACK_ORIGIN: "http://127.0.0.1:3210",
};
writeFileSync(envPath, Object.entries(values).map(([name, value]) => `${name}=${value}`).join("\n") + "\n", { mode: 0o600 });
console.log("Created local-only configuration and two private QA credentials. No values are printed. Run Next from the repository root.");
