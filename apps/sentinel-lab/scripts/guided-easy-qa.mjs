import { createRequire } from "node:module";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = resolve(import.meta.dirname, "../../..");
const base = process.env.EIDOS_QA_URL || "http://127.0.0.1:3210";
const access = process.env.EIDOS_QA_ACCESS_FILE
  ? JSON.parse(readFileSync(process.env.EIDOS_QA_ACCESS_FILE, "utf8")).url : null;
const keys = JSON.parse(readFileSync(resolve(root, "artifacts/sentinel-guided-private/local-access.json"), "utf8"));
const output = resolve(root, `artifacts/sentinel-guided-20260908/${process.env.EIDOS_QA_TAG || "easy-browser"}`);
mkdirSync(output, { recursive: true });

const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: "reduce", acceptDownloads: true });
const page = await context.newPage();
const errors = [];
page.on("pageerror", error => errors.push(error.message));
const receipt = { base, browser: "Playwright Chromium", startedAt: new Date().toISOString(), checks: [], screenshots: [], errors };
const screenshot = async name => { await page.screenshot({ path: resolve(output, `${name}.png`), fullPage: true }); receipt.screenshots.push(`${name}.png`); };
const check = async (name, operation) => { await operation(); receipt.checks.push({ name, status: "passed" }); };

try {
  if (access) await page.goto(access, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.goto(base, { waitUntil: "networkidle", timeout: 60000 });
  await page.getByRole("heading", { name: /From your data.*to a clearer answer/ }).waitFor();
  await screenshot("01-easy-entry");

  await check("private sign-in and real CSV upload", async () => {
    await page.getByLabel("Access key", { exact: true }).fill(keys.alice);
    await page.getByRole("button", { name: "Sign in", exact: true }).click();
    await page.getByRole("button", { name: "Sign out" }).waitFor();
    await page.getByLabel("Data file").setInputFiles(resolve(root, "artifacts/sentinel-guided-20260908/fixtures/service-latency.csv"));
    await page.getByRole("heading", { name: "Check its meaning", exact: true }).waitFor({ timeout: 240000 });
  });
  await screenshot("02-confirm-data");

  await check("Easy/Advanced retains data interpretation and excludes labels", async () => {
    await page.getByLabel("What does one row represent?").fill("One synthetic service measurement per minute");
    await page.getByLabel("Time column").selectOption("timestamp");
    await page.getByLabel("Separate machines, services or people by").selectOption("host");
    await page.getByRole("button", { name: "Advanced settings" }).click();
    if (await page.getByLabel("What does one row represent?").inputValue() !== "One synthetic service measurement per minute") throw Error("Switch discarded interpretation");
    if (!await page.locator(".el-measurements label").filter({ hasText: "Evaluation label — excluded" }).locator("input[type=checkbox]").isDisabled()) throw Error("Evaluation label selectable as a feature");
    await page.getByLabel("Time zone when offsets are absent").fill("UTC");
    await page.getByLabel("latency_ms units").fill("ms");
    await page.getByLabel("load units").fill("%");
    await page.getByRole("button", { name: "Confirm this interpretation" }).click();
    await page.getByRole("heading", { name: "Run experiment", exact: true }).waitFor({ timeout: 240000 });
  });

  await check("forecast contract survives setup switch", async () => {
    await page.getByRole("radio", { name: /Forecast a measurement/ }).check();
    await page.getByLabel("Measurement to forecast").selectOption("latency_ms");
    await page.getByLabel("How far ahead? (seconds)").fill("60");
    await page.getByLabel("Maximum late-match window (seconds)").fill("60");
    await page.getByRole("button", { name: "Easy setup" }).click();
    if (await page.getByLabel("How far ahead? (seconds)").inputValue() !== "60") throw Error("Switch discarded forecast horizon");
    await page.getByRole("button", { name: "Advanced settings" }).click();
    if (await page.getByLabel("Maximum late-match window (seconds)").inputValue() !== "60") throw Error("Switch discarded late-match window");
  });
  await screenshot("03-forecast-contract");

  await check("actual Torch run, durable result and original input hash", async () => {
    await page.getByRole("button", { name: "Run experiment", exact: true }).last().click();
    await page.getByRole("heading", { name: "Understand results", exact: true }).waitFor({ timeout: 240000 });
    const pending = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download full result" }).click();
    const download = await pending;
    await download.saveAs(resolve(output, "easy-result.json"));
    const result = JSON.parse(readFileSync(resolve(output, "easy-result.json"), "utf8"));
    if (result.engine?.class !== "RLS_Reservoir" || !result.inputSha256 || !result.forecasts?.length) throw Error("Completed result lacks engine, source hash or issued forecasts");
    receipt.runId = result.id; receipt.datasetId = result.datasetId; receipt.engine = result.engine; receipt.metrics = result.metrics;
  });
  await screenshot("04-results-desktop");

  await check("narrow viewport, keyboard and reduced motion", async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await screenshot("05-results-mobile");
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)) throw Error("Mobile document overflow");
    await page.keyboard.press("Control+Home"); await page.keyboard.press("Tab");
    if (!await page.evaluate(() => document.activeElement !== document.body)) throw Error("No keyboard focus");
    if (!await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)) throw Error("Reduced motion not honored");
  });

  await check("saved result reopens after reload", async () => {
    await page.reload({ waitUntil: "networkidle" });
    await page.locator("#saved-work summary").click();
    await page.getByRole("button", { name: /Causal canonical Torch/ }).first().click();
    await page.getByRole("heading", { name: "Understand results", exact: true }).waitFor();
    await page.locator("details").filter({ hasText: "Technical metrics, limitations and reproducibility" }).locator("summary").click();
    await page.getByText(/Original input SHA-256/).waitFor();
  });

  await check("second owner cannot read saved dataset or run", async () => {
    const bob = await browser.newContext();
    if (access) await bob.request.get(access);
    for (const path of [`datasets/${receipt.datasetId}`, `datasets/${receipt.datasetId}/source`, `runs/${receipt.runId}`, `runs/${receipt.runId}/download`]) {
      const response = await bob.request.get(`${base}/api/lab/v1/${path}`, { headers: { Authorization: `Bearer ${keys.bob}` } });
      if (response.status() !== 404) throw Error(`${path} returned ${response.status()} for another owner`);
    }
    await bob.close();
  });

  await check("sign-out clears private result", async () => {
    await page.getByRole("button", { name: "Sign out" }).click();
    await page.getByLabel("Access key", { exact: true }).waitFor();
    if (await page.getByText(/Original input SHA-256/).count()) throw Error("Private result remained after sign-out");
  });
  if (errors.length) throw Error(`Browser errors: ${errors.join("; ")}`);
  receipt.status = "passed";
} catch (error) {
  receipt.status = "failed"; receipt.failure = error.message;
  await screenshot("failure").catch(() => undefined);
  process.exitCode = 1;
} finally {
  receipt.finishedAt = new Date().toISOString();
  writeFileSync(resolve(output, "easy-browser-receipt.json"), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify({ status: receipt.status, checks: receipt.checks, failure: receipt.failure, runId: receipt.runId }, null, 2));
  await browser.close();
}
