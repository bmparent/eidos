import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const pw = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const source = process.env.PG_COMPILED_PACKAGE_DIR;
assert.ok(source, 'PG_COMPILED_PACKAGE_DIR must name the actual Next-generated package directory');
const out = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/i, '$1'));
const archive = await readFile(path.join(source, 'compiled-server-approved.zip'));
const compileReceipt = JSON.parse(await readFile(path.join(source, 'receipt.json'), 'utf8'));
const archiveSha256 = createHash('sha256').update(archive).digest('hex');
assert.equal(archiveSha256, compileReceipt.archiveSha256);
assert.equal(archive.length, compileReceipt.archiveBytes);
const records = [];
for (const engine of ['chromium', 'firefox', 'webkit']) {
  const browser = await pw[engine].launch({ headless: true });
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    const page = await context.newPage(), record = { engine, width, errors: [], console: [], requests: [], checks: [] };
    records.push(record);
    page.on('pageerror', error => record.errors.push(error.message));
    page.on('console', message => { if (['warning', 'error'].includes(message.type())) record.console.push({ type: message.type(), text: message.text() }); });
    await context.route(/^https?:/, route => { record.requests.push(route.request().url()); return route.abort(); });
    await page.goto(pathToFileURL(path.join(source, 'site/index.html')).href);
    await page.locator('h1').first().waitFor();
    assert.ok((await page.locator('body').innerText()).includes('Recovered Atlas'));
    record.checks.push('Actual compiled Next package renders independent content');
    if (width === 390) {
      const menu = page.getByRole('button', { name: 'Toggle navigation', exact: true });
      await menu.click(); assert.equal(await menu.getAttribute('aria-expanded'), 'true');
      await page.locator('#page-nav a').first().click(); assert.equal(await menu.getAttribute('aria-expanded'), 'false');
      record.checks.push('Actual compiled package runtime opens and closes mobile navigation');
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    assert.deepEqual(record.errors, []); assert.deepEqual(record.console, []); assert.deepEqual(record.requests, []);
    await mkdir(out, { recursive: true });
    await page.screenshot({ path: path.join(out, `${engine}-${width}.png`), fullPage: true });
    console.log(`${engine} ${width}: PASS actual Next package independent runtime`);
    await context.close();
  }
  await browser.close();
}
await copyFile(path.join(source, 'receipt.json'), path.join(out, 'compiled-receipt.json'));
await writeFile(path.join(out, 'receipt.json'), JSON.stringify({ timestampUtc: new Date().toISOString(), source: 'Actual compiled Next production module, plain Node SQLite freeze/read', archiveSha256, archiveBytes: archive.length, network: 'Blocked all HTTP(S); file URLs only', physicalDevice: false, records }, null, 2) + '\n');
