import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const artifactsDir = join(root, "artifacts");
const chromeBin = process.env.CHROME_BIN;
const profileDir = mkdtempSync(join(tmpdir(), "eidos-sentinel-cdp-"));
const appUrl = "http://127.0.0.1:3100";
const cdpUrl = "http://127.0.0.1:9333";
mkdirSync(artifactsDir, { recursive: true });

if (!chromeBin) throw new Error("Set CHROME_BIN to a Chrome or chrome-headless-shell executable.");

const server = spawn(process.execPath, [join(root, "node_modules/next/dist/bin/next"), "start", "--hostname", "127.0.0.1", "--port", "3100"], {
  cwd: root,
  env: process.env,
  stdio: ["ignore", "pipe", "pipe"],
  detached: true,
});
const chrome = spawn(chromeBin, [
  "--headless",
  "--no-sandbox",
  "--disable-gpu",
  "--disable-dev-shm-usage",
  `--user-data-dir=${profileDir}`,
  "--remote-debugging-port=9333",
  "about:blank",
], { stdio: ["ignore", "pipe", "pipe"], detached: true });

let serverLog = "";
let chromeLog = "";
server.stdout.on("data", (chunk) => { serverLog += chunk; });
server.stderr.on("data", (chunk) => { serverLog += chunk; });
chrome.stdout.on("data", (chunk) => { chromeLog += chunk; });
chrome.stderr.on("data", (chunk) => { chromeLog += chunk; });

const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));

async function waitFor(url, timeout = 25_000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
    } catch {
      // Expected while the process is starting.
    }
    await delay(125);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

class CdpClient {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 0;
    this.pending = new Map();
    this.handlers = new Map();
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const waiter = this.pending.get(message.id);
        if (!waiter) return;
        this.pending.delete(message.id);
        if (message.error) waiter.reject(new Error(message.error.message));
        else waiter.resolve(message.result);
        return;
      }
      for (const handler of this.handlers.get(message.method) ?? []) handler(message.params);
    });
  }

  send(method, params = {}) {
    const id = ++this.nextId;
    return new Promise((resolveSend, rejectSend) => {
      this.pending.set(id, { resolve: resolveSend, reject: rejectSend });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  on(method, handler) {
    this.handlers.set(method, [...(this.handlers.get(method) ?? []), handler]);
  }
}

async function connect(webSocketUrl) {
  const socket = new WebSocket(webSocketUrl);
  await new Promise((resolveOpen, rejectOpen) => {
    socket.addEventListener("open", resolveOpen, { once: true });
    socket.addEventListener("error", rejectOpen, { once: true });
  });
  return new CdpClient(socket);
}

async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || "Browser evaluation failed");
  return result.result.value;
}

async function waitForExpression(client, expression, timeout = 15_000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await evaluate(client, expression)) return;
    await delay(100);
  }
  throw new Error(`Timed out waiting for browser expression: ${expression}`);
}

async function load(client, url) {
  await client.send("Page.navigate", { url });
  await waitForExpression(client, "document.readyState === 'complete' && document.body.innerText.trim().length > 0");
  await delay(250);
}

async function setViewport(client, width, height, mobile = false) {
  await client.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile, screenWidth: width, screenHeight: height });
}

async function screenshot(client, filename) {
  const metrics = await client.send("Page.getLayoutMetrics");
  const size = metrics.cssContentSize ?? metrics.contentSize;
  const result = await client.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: size.width, height: size.height, scale: 1 },
  });
  const path = join(artifactsDir, filename);
  writeFileSync(path, Buffer.from(result.data, "base64"));
  return path;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

let client;
try {
  await Promise.all([waitFor(appUrl), waitFor(`${cdpUrl}/json/version`)]);
  const target = await fetch(`${cdpUrl}/json/new?${encodeURIComponent(appUrl)}`, { method: "PUT" }).then((response) => response.json());
  client = await connect(target.webSocketDebuggerUrl);
  const consoleErrors = [];
  client.on("Runtime.exceptionThrown", (params) => consoleErrors.push(params.exceptionDetails?.text ?? "Runtime exception"));
  client.on("Log.entryAdded", (params) => { if (params.entry?.level === "error") consoleErrors.push(params.entry.text); });
  client.on("Runtime.consoleAPICalled", (params) => { if (params.type === "error") consoleErrors.push(params.args?.map((arg) => arg.value ?? arg.description).join(" ")); });
  await Promise.all([client.send("Page.enable"), client.send("Runtime.enable"), client.send("Log.enable"), client.send("Network.enable")]);

  await setViewport(client, 1440, 1000);
  await load(client, appUrl);
  await delay(750);
  const initial = await evaluate(client, `({
    title: document.title,
    text: document.body.innerText,
    overlays: document.querySelectorAll('[data-nextjs-dialog], .vite-error-overlay, #webpack-dev-server-client-overlay').length,
    h1: document.querySelector('h1')?.innerText,
    selects: document.querySelectorAll('select').length
  })`);
  assert(initial.title === "Eidos / Sentinel Lab", "Unexpected page title");
  assert(initial.h1?.includes("Proof is blocked"), "Hero did not render");
  assert(initial.text.replace(/\s/g, "").includes("BLOCKED_RESOURCE_BEFORE_HELDOUT"), "Verdict is missing");
  assert(initial.text.includes("Engineering smoke does not advance proof gates"), "Proof disclaimer is missing");
  assert(initial.selects === 3, "Expected three configuration selectors");
  assert(initial.overlays === 0, "Framework error overlay detected");
  const desktopPath = await screenshot(client, "sentinel-lab-desktop.png");

  await evaluate(client, `(() => {
    const selects = document.querySelectorAll('.config-rail select');
    const values = ['S3_regime_shift', '1', '480'];
    const setValue = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
    selects.forEach((select, index) => {
      setValue.call(select, values[index]);
      select.dispatchEvent(new Event('input', { bubbles: true }));
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    return true;
  })()`);
  await delay(350);
  await evaluate(client, `(() => { document.querySelector('.run-button').click(); return true; })()`);
  try {
    await waitForExpression(client, "document.querySelector('.trace-heading .eyebrow')?.innerText.toLowerCase().includes('eng-s3_regime_shift-s1-f480')");
  } catch (interactionError) {
    const diagnostic = await evaluate(client, `({
      selections: [...document.querySelectorAll('.config-rail select')].map((select) => select.value),
      heading: document.querySelector('.trace-heading .eyebrow')?.innerText,
      error: document.querySelector('.error-banner')?.innerText,
      button: document.querySelector('.run-button')?.innerText,
      disabled: document.querySelector('.run-button')?.disabled
    })`);
    throw new Error(`${interactionError.message}; state=${JSON.stringify(diagnostic)}`);
  }
  assert(await evaluate(client, "!document.querySelector('.run-button').disabled"), "Run button remained disabled");

  const errorsBeforeHeldOutCheck = consoleErrors.length;
  const heldOut = await evaluate(client, `fetch('/api/smoke', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({scenario:'S0_nominal', seed:100, frames:240, system:'eidos_ms_v1_observer'})
  }).then(async (response) => ({status: response.status, body: await response.json()}))`);
  assert(heldOut.status === 400, "Held-out seed request was not rejected");
  assert(heldOut.body.error.includes("engineering seeds 0 and 1"), "Held-out rejection was not explicit");
  const expectedNetworkErrors = consoleErrors.splice(errorsBeforeHeldOutCheck);
  assert(expectedNetworkErrors.every((message) => message.includes("400")), `Unexpected error during negative API check: ${expectedNetworkErrors.join(" | ")}`);

  await evaluate(client, `(() => { [...document.querySelectorAll('.primary-nav button')].find((button) => button.innerText.trim().toLowerCase() === 'gates').click(); return true; })()`);
  await waitForExpression(client, "document.querySelectorAll('.gate-list article').length === 7");
  assert(await evaluate(client, "[...document.querySelectorAll('.locked-badge')].every((node) => node.innerText === 'LOCKED')"), "A proof gate appeared unlocked");
  await evaluate(client, `(() => { [...document.querySelectorAll('.primary-nav button')].find((button) => button.innerText.trim().toLowerCase() === 'compare').click(); return true; })()`);
  await waitForExpression(client, "document.querySelectorAll('.comparison-table tbody tr').length === 4");

  await setViewport(client, 390, 844, true);
  await load(client, appUrl);
  const mobile = await evaluate(client, `({
    bodyWidth: document.body.scrollWidth,
    viewport: innerWidth,
    menuVisible: getComputedStyle(document.querySelector('.menu-button')).display !== 'none',
    desktopEvidenceVisible: getComputedStyle(document.querySelector('.desktop-evidence')).display !== 'none'
  })`);
  assert(mobile.bodyWidth <= mobile.viewport, `Mobile layout overflows: ${mobile.bodyWidth}px / ${mobile.viewport}px`);
  assert(mobile.menuVisible, "Mobile menu control is hidden");
  assert(!mobile.desktopEvidenceVisible, "Desktop evidence panel remains visible on mobile");
  const mobilePath = await screenshot(client, "sentinel-lab-mobile.png");
  await evaluate(client, "document.querySelector('.menu-button').click()");
  await waitForExpression(client, "getComputedStyle(document.querySelector('.primary-nav')).display === 'grid'");

  const finalOverlayCount = await evaluate(client, "document.querySelectorAll('[data-nextjs-dialog], .vite-error-overlay, #webpack-dev-server-client-overlay').length");
  assert(finalOverlayCount === 0, "Framework error overlay appeared during interactions");
  assert(consoleErrors.length === 0, `Browser console errors: ${consoleErrors.join(" | ")}`);

  const report = {
    status: "PASS",
    desktopPath,
    mobilePath,
    checks: {
      meaningfulContent: true,
      noFrameworkOverlay: true,
      noConsoleErrors: true,
      smokeRunInteraction: true,
      heldOutSeedRejected: true,
      allSevenGatesLocked: true,
      compareRows: 4,
      mobileNoPageOverflow: true,
      mobileMenuOpens: true,
    },
  };
  writeFileSync(join(artifactsDir, "browser-qa.json"), JSON.stringify(report, null, 2));
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : error}\n`);
  process.stderr.write(`Server log:\n${serverLog.slice(-4000)}\nChrome log:\n${chromeLog.slice(-4000)}\n`);
  process.exitCode = 1;
} finally {
  if (client?.socket) client.socket.close();
  for (const child of [server, chrome]) {
    if (!child.pid) continue;
    try { process.kill(-child.pid, "SIGKILL"); } catch { child.kill("SIGKILL"); }
  }
  await delay(150);
  rmSync(profileDir, { recursive: true, force: true });
}
