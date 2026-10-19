// Plain Node: deliberately no tsx/esbuild transpilation of the tested renderer.
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { DatabaseSync } = require('node:sqlite');
const vm = require('node:vm');
const app = path.resolve('apps/sentinel-lab');
const outArg = process.argv.find(value => value.startsWith('--out='));
const out = path.resolve(outArg ? outArg.slice('--out='.length) : 'artifacts/playground-unified-backend-20261014/continuation-final/compiled-current-reference');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
function entries(bytes) {
  const files = new Map(); let offset = 0;
  while (offset + 30 <= bytes.length && bytes.readUInt32LE(offset) === 0x04034b50) {
    const size = bytes.readUInt32LE(offset + 18), names = bytes.readUInt16LE(offset + 26), extra = bytes.readUInt16LE(offset + 28);
    if (bytes.readUInt16LE(offset + 8) !== 0 || extra) throw Error('Use an uncompressed controlled snapshot ZIP.');
    const start = offset + 30 + names;
    if (start + size > bytes.length) throw Error('Truncated ZIP');
    files.set(bytes.subarray(offset + 30, start).toString(), bytes.subarray(start, start + size)); offset = start + size;
  }
  return files;
}
class Query {
  constructor(sql, text, values = []) { Object.assign(this, { sql, text, values }); }
  bind(...values) { return new Query(this.sql, this.text, values); }
  async first() { return this.sql.prepare(this.text).get(...this.values) || null; }
  async all() { return { results: this.sql.prepare(this.text).all(...this.values) }; }
  async run() { return { meta: { changes: Number(this.sql.prepare(this.text).run(...this.values).changes) } }; }
}
(async () => {
  if (!process.argv[2]) throw Error('Supply a controlled frozen ZIP containing SNAPSHOT.json.');
  fs.mkdirSync(out, { recursive: true });
  const sourceZip = fs.readFileSync(process.argv[2]);
  const snapshot = JSON.parse(entries(sourceZip).get('SNAPSHOT.json'));
  const route = path.join(app, '.next/server/app/api/works/v1/[...path]/route.js');
  globalThis.fetch = async () => { throw Error('Compiled renderer attempted outbound traffic'); };
  let moduleId, moduleChunk;
  for (const match of fs.readFileSync(route, 'utf8').matchAll(/R\.c\("([^"]+)"\)/g)) {
    const filename = path.join(app, '.next', match[1]), loaded = require(filename);
    if (!Array.isArray(loaded)) continue;
    for (const factory of loaded.filter(value => typeof value === 'function'))
      for (const found of factory.toString().matchAll(/\.s\(\[([^\]]*)\],(\d+)\)/g))
        if (found[1].includes('"freezeProjectDelivery"')) { moduleId = Number(found[2]); moduleChunk = filename; }
  }
  if (!moduleId) throw Error('Compiled delivery export was not found in the actual route graph.');
  require(route); // Loads the exact optimized production chunks used by the Next route.
  const runtime = require(path.join(app, '.next/server/chunks/[turbopack]_runtime.js'))('server/app/api/works/v1/[...path]/route.js');
  const delivery = runtime.m(moduleId).exports;
  if (typeof delivery.freezeProjectDelivery !== 'function') throw Error('Unexpected production module identity; inspect the rebuilt chunks.');
  const sql = new DatabaseSync(':memory:'); sql.exec('PRAGMA foreign_keys=ON');
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql']) sql.exec(fs.readFileSync(path.join(app, 'lib/works/vendor/migrations', name), 'utf8'));
  sql.prepare('INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,?,?,?)').run('compiled-probe', 'probe@example.test', 'compiled-probe', 'person', '2026-10-14', '2026-10-14');
  const database = { prepare: text => new Query(sql, text), async batch(statements) {
    sql.exec('BEGIN'); try { const results = []; for (const statement of statements) results.push(await statement.run()); sql.exec('COMMIT'); return results; }
    catch (error) { sql.exec('ROLLBACK'); throw error; }
  } };
  const env = { EIDOS_RUNTIME: 'sentinel', EIDOS_DB: database, EIDOS_LOCAL_TEST: 'true', PUBLIC_SITE_URL: 'http://localhost:8788', EIDOS_PLAYGROUND_DELIVERY_MODE: 'test' };
  const saved = await delivery.freezeProjectDelivery(env, 'compiled-probe', { requestId: 'compiled-production-probe', snapshot });
  const recovered = await delivery.readProjectDeliveryArchive(env, 'compiled-probe', saved.id);
  const archive = Buffer.from(recovered.archive), files = entries(archive), script = files.get('script.js');
  if (!script) throw Error('No exported script');
  new vm.Script(script.toString(), { filename: 'compiled-server-script.js' });
  if (/\b__name\b/.test(script.toString())) throw Error('Production exported script leaks __name');
  for (const [name, bytes] of files) {
    if (name.includes('..') || path.isAbsolute(name)) throw Error('Unsafe fixture export path');
    const target = path.join(out, 'site', name); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, bytes);
  }
  fs.writeFileSync(path.join(out, 'compiled-server-approved.zip'), archive);
  const chunk = fs.readFileSync(moduleChunk);
  const receipt = { timestampUtc: new Date().toISOString(), build: 'Next 16.3.8 optimized production build, actual route module ' + moduleId, moduleChunk: path.relative(app, moduleChunk).replaceAll('\\', '/'), transpilerHarness: 'Plain Node; no tsx/esbuild loader', routeSha256: hash(fs.readFileSync(route)), chunkSha256: hash(chunk), inputZipSha256: hash(sourceZip), snapshotHash: snapshot.snapshotHash, archiveSha256: hash(archive), archiveBytes: archive.length, byteIdenticalToInput: archive.equals(sourceZip), scriptSha256: hash(script), scriptBytes: script.length, nameHelperLeak: false, scriptParse: 'passed', storageRoundTrip: 'Actual compiled freeze/read functions with real in-memory SQLite', browserExecution: 'pending independent archive opening', network: 'fetch forbidden', files: [...files.keys()] };
  fs.writeFileSync(path.join(out, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
  sql.close(); console.log(JSON.stringify(receipt));
  if (process.argv.includes('--require-match') && !receipt.byteIdenticalToInput) throw Error('Actual production Next archive differs from the approved input ZIP.');
})().catch(error => { console.error(error); process.exitCode = 1; });
