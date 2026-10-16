const { createRequire } = require('node:module');
const fs = require('node:fs');
const path = require('node:path');
const frontend = process.argv[2];
if (!frontend) throw Error('Supply the authoritative frontend repository root.');
const { build } = createRequire(path.resolve(frontend, 'package.json'))('esbuild');
(async () => {
  const zip = fs.readFileSync('artifacts/playground-unified-backend-20261014/delivery/compiled-runtime-before/fixture-input.zip');
  let offset = 0, snapshot;
  while (offset + 30 <= zip.length && zip.readUInt32LE(offset) === 0x04034b50) {
    const names = zip.readUInt16LE(offset + 26), extra = zip.readUInt16LE(offset + 28), length = zip.readUInt32LE(offset + 18), start = offset + 30 + names + extra;
    if (zip.subarray(offset + 30, offset + 30 + names).toString() === 'SNAPSHOT.json') snapshot = JSON.parse(zip.subarray(start, start + length));
    offset = start + length;
  }
  const code = await build({ stdin: { contents: 'export {buildFrozenPackage} from "./src/playground/purchaseSnapshot";', resolveDir: path.resolve(frontend) }, bundle: true, format: 'esm', platform: 'node', write: false, keepNames: true, minify: false });
  const { buildFrozenPackage } = await import('data:text/javascript;base64,' + Buffer.from(code.outputFiles[0].text).toString('base64'));
  const frozen = await buildFrozenPackage(snapshot);
  fs.writeFileSync('apps/sentinel-lab/.vercel/compiled-delivery-check/canonical-reference.zip', frozen.archive);
  console.log(JSON.stringify({ snapshotHash: snapshot.snapshotHash, referenceArchiveSha256: frozen.receipt.archiveSha256, archiveBytes: frozen.archive.length, build: 'esbuild node ESM keepNames=true minify=false' }));
})().catch(error => { console.error(error); process.exitCode = 1; });
