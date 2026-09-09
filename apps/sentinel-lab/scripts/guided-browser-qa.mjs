import { createRequire } from "node:module";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = resolve(import.meta.dirname, "../../..");
const output = resolve(root, `artifacts/sentinel-guided-20260908/${process.env.EIDOS_QA_TAG || "browser"}`);
mkdirSync(output, { recursive: true });
const base = process.env.EIDOS_QA_URL || "http://127.0.0.1:3210";
const access = process.env.EIDOS_QA_ACCESS_FILE ? JSON.parse(readFileSync(process.env.EIDOS_QA_ACCESS_FILE, "utf8")).url : null;
const keys = JSON.parse(readFileSync(resolve(root, "artifacts/sentinel-guided-private/local-access.json"), "utf8"));
const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: "reduce" });
const page = await context.newPage();
const errors = [];
page.on("pageerror", error => errors.push(error.message));
const receipt = { base, browser: "Playwright Chromium", reason: "Browser plugin not available", startedAt: new Date().toISOString(), checks: [], screenshots: [], errors };
const screenshot = async name => { await page.screenshot({ path: resolve(output, `${name}.png`), fullPage: true }); receipt.screenshots.push(`${name}.png`); };
async function check(name, operation) { await operation(); receipt.checks.push({ name, status: "passed" }); }
try {
  await page.goto("https://eidos-sentinel-lab.vercel.app", { waitUntil: "domcontentloaded", timeout: 60000 });
  await screenshot("before-production");
  receipt.productionBefore = { title: await page.title(), url: page.url() };
  if (access) await page.goto(access, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.goto(base, { waitUntil: "networkidle", timeout: 60000 });
  await page.getByRole("heading", { name: "Understand what changed." }).waitFor();
  await screenshot("01-add-data-desktop");
  await check("approved pilot login", async () => { await page.getByLabel("Access key", { exact: true }).fill(keys.alice); await page.getByRole("button", { name: "Sign in", exact: true }).click(); await page.getByText("Signed in · pilot access").waitFor(); });
  await check("real browser CSV upload and parser completion", async () => {
    await page.getByLabel("Upload data file").setInputFiles(resolve(root, "artifacts/sentinel-guided-20260908/fixtures/service-latency.csv"));
    await page.getByRole("heading", { name: "Confirm understanding", exact: true }).waitFor({ timeout: 240000 });
  });
  await screenshot("02-confirm-schema");
  await check("confirmed chronology, entity, units and interpretation", async () => {
    await page.getByLabel("One row represents", { exact: true }).fill("One synthetic service latency measurement per minute");
    await page.getByLabel("Timestamp", { exact: true }).selectOption("timestamp");
    await page.getByLabel("Entity key", { exact: true }).selectOption("host");
    await page.getByLabel("latency_ms units").fill("ms");
    await page.getByLabel("load units").fill("%");
    await page.getByRole("button", { name: "Confirm interpretation" }).click();
    await page.getByRole("heading", { name: "Analyze", exact: true }).waitFor({ timeout: 240000 });
  });
  await screenshot("03-analyze");
  await check("actual Torch analysis through persisted result", async () => {
    await page.getByRole("button", { name: "Run analysis" }).click();
    await page.getByRole("heading", { name: "Investigate", exact: true }).waitFor({ timeout: 240000 });
    await page.getByText("Your data · actual Torch Eidos + identified baseline", { exact: true }).waitFor();
  });
  await screenshot("04-investigate-desktop");
  await check("original record drill-down", async () => {
    await page.getByRole("button", { name: /^View r/ }).first().click();
    await page.getByText(/Source SHA-256/).waitFor();
  });
  await check("grounded result question", async () => {
    await page.getByRole("button", { name: "Find supporting evidence" }).click();
    await page.getByText("deterministic finding/evidence query; no language-model inference").waitFor();
  });
  await check("review saved separately from frozen output", async () => {
    await page.getByLabel(/^Review finding-/).first().selectOption("useful");
    await page.getByText(/Review saved with provenance/).waitFor();
  });
  await screenshot("05-evidence-and-question");
  await check("download authenticated result", async () => {
    const pending = page.waitForEvent("download"); await page.getByRole("button", { name: "Download results" }).click();
    const download = await pending; await download.saveAs(resolve(output, "browser-result.json"));
    const result = JSON.parse(readFileSync(resolve(output, "browser-result.json"), "utf8"));
    if (result.engine?.class !== "RLS_Reservoir" || !result.inputSha256 || !result.forecasts.length) throw Error("Missing actual engine/input/forecast evidence");
    receipt.runId = result.id; receipt.datasetId = result.datasetId; receipt.engine = result.engine; receipt.metrics = result.metrics;
  });
  await page.setViewportSize({ width: 390, height: 844 }); await screenshot("06-investigate-mobile");
  await check("mobile layout has no document overflow", async () => {
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
    if (overflow) throw Error("Mobile document overflow");
  });
  await check("keyboard navigation and reduced motion", async () => {
    await page.keyboard.press("Control+Home"); await page.keyboard.press("Tab");
    const hasFocus = await page.evaluate(() => document.activeElement !== document.body);
    if (!hasFocus) throw Error("No keyboard focus");
    if (!await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)) throw Error("Reduced motion missing");
  });
  await page.reload({ waitUntil: "networkidle" });
  await check("refresh and saved-run recovery", async () => {
    await page.getByRole("button", { name: "Saved work", exact: true }).click();
    await page.getByRole("button", { name: /Causal canonical Torch/ }).first().click();
    await page.getByRole("heading", { name: "Investigate", exact: true }).waitFor();
  });
  await screenshot("07-saved-run-mobile");
  await check("second user cannot access every first-user resource", async () => {
    const bob = await browser.newContext();
    if (access) await bob.request.get(access);
    for (const path of [`datasets/${receipt.datasetId}`, `datasets/${receipt.datasetId}/source`, `runs/${receipt.runId}`, `runs/${receipt.runId}/download`, `jobs/${receipt.runId}`]) {
      const response = await bob.request.get(`${base}/api/lab/v1/${path}`, { headers: { Authorization: `Bearer ${keys.bob}` } });
      if (response.status() !== 404) throw Error(`${path} returned ${response.status()} for second user`);
    }
    await bob.close();
  });
  if (errors.length) throw Error(`Browser runtime errors: ${errors.join("; ")}`);
  receipt.status = "passed";
} catch (error) {
  receipt.status = "failed"; receipt.failure = error.message;
  await screenshot("failure").catch(() => undefined);
  process.exitCode = 1;
} finally {
  receipt.finishedAt = new Date().toISOString();
  writeFileSync(resolve(output, "browser-receipt.json"), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify({ status: receipt.status, checks: receipt.checks, failure: receipt.failure, metrics: receipt.metrics }, null, 2));
  await browser.close();
}
