import { createHash } from 'node:crypto';
import { lstat, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export async function verifyPrivateTemplateArchives(root, manifest) {
  if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.archives) || manifest.archives.length < 8) throw Error('Missing private template archive inventory.');
  const seen = new Set();
  for (const item of manifest.archives) {
    if (!/^templates\/(switchboard|tideglass|matter|nightjar)\/\d+\.\d+\.\d+\/(developer|wordpress)\.zip$/.test(item.archiveKey) || !/^[a-f0-9]{64}$/.test(item.sha256) || !Number.isSafeInteger(item.bytes) || item.bytes <= 0 || seen.has(item.archiveKey)) throw Error('Invalid private template archive inventory.');
    seen.add(item.archiveKey);
    const filename = path.resolve(root, item.archiveKey);
    if (!filename.startsWith(path.resolve(root) + path.sep) || !(await lstat(filename)).isFile() || (await lstat(filename)).isSymbolicLink()) throw Error('Unsafe private template archive: ' + item.archiveKey);
    const bytes = await readFile(filename);
    if (bytes.length !== item.bytes || createHash('sha256').update(bytes).digest('hex') !== item.sha256) throw Error('Private template archive differs from frozen inventory: ' + item.archiveKey);
  }
  return seen.size;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const active = process.env.EIDOS_TEMPLATE_TEST_ENABLED === 'true' || process.env.EIDOS_TEMPLATE_LIVE_ENABLED === 'true' || Boolean(process.env.EIDOS_TEMPLATE_SOURCE_REVISION);
  if (active) {
    const root = path.resolve('private');
    const manifest = JSON.parse(await readFile(path.join(root, 'templates-manifest.json'), 'utf8'));
    const count = await verifyPrivateTemplateArchives(root, manifest);
    const { templateProducts } = await import('../lib/works/vendor/functions/_shared/platform/templateCatalog.ts');
    for (const product of templateProducts) {
      const key = product.archiveKey;
      if (!manifest.archives.some(item => item.archiveKey === key)) throw Error('Current catalog archive is missing: ' + key);
    }
    console.log(`Verified ${count} current and retained private template archives before build.`);
  } else console.log('Template commerce is disabled; private archive gate is inactive.');
}
