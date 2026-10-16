import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, lstat, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { verifyPrivateTemplateArchives } from './check-template-archives.mjs';

// A source-only Git deployment cannot include the ignored purchased archives.
// This prepares a complete protected TEST bundle without reading credentials.
const outputArg = process.argv.find(value => value.startsWith('--out='));
if (!outputArg) throw Error('Supply --out=<private deployment request JSON>.');
const root = process.cwd(), app = 'apps/sentinel-lab/';
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
const sha = git('rev-parse', 'HEAD');
const environmentBranchArg = process.argv.find(value => value.startsWith('--environment-branch='));
const environmentBranch = environmentBranchArg ? environmentBranchArg.slice('--environment-branch='.length) : 'codex/paid-templates-20261010';
if (!['codex/paid-templates-20261010', 'codex/playground-unified-20261014'].includes(environmentBranch)) throw Error('Unapproved isolated TEST environment branch.');
if (environmentBranch === 'codex/playground-unified-20261014' && git('branch', '--show-current') !== environmentBranch) throw Error('Unified Playground bundle must come from its matching isolated branch.');
if (JSON.parse(await readFile(app + 'package.json', 'utf8')).name !== 'eidos-sentinel-lab') throw Error('Run from the backend repository root.');
const manualOnly = process.argv.includes('--manual-only');
if (manualOnly && environmentBranch !== 'codex/playground-unified-20261014') throw Error('Manual-only bundles are restricted to the isolated Playground branch.');
if (manualOnly && git('status', '--porcelain')) throw Error('Commit the reviewed manual-only source before bundling.');
const manifest = manualOnly ? { archives: [] } : JSON.parse(await readFile(app + 'private/templates-manifest.json', 'utf8'));
const count = manualOnly ? 0 : await verifyPrivateTemplateArchives(path.resolve(app, 'private'), manifest);
const selected = new Set(git('ls-files', '-co', '--exclude-standard', '--', app).split('\n').filter(Boolean));
if (!manualOnly) selected.add(app + 'private/templates-manifest.json');
for (const item of manifest.archives) selected.add(app + 'private/' + item.archiveKey);
const files = [], inventory = [];
for (const filename of [...selected].sort()) {
  if (!filename.startsWith(app)) throw Error('Unexpected bundle scope.');
  if (filename.split('/').some(part => part.startsWith('.env') || ['.next', '.vercel', 'node_modules', 'tests', 'docs', 'artifacts'].includes(part)) || /\.(log|tsbuildinfo)$/.test(filename)) continue;
  if (!(await lstat(filename)).isFile() || (await lstat(filename)).isSymbolicLink()) throw Error('Unsafe bundle path: ' + filename);
  const bytes = await readFile(filename);
  files.push({ file: filename, data: bytes.toString('base64'), encoding: 'base64' });
  inventory.push({ file: filename, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
}
const bundleHash = createHash('sha256').update(JSON.stringify(inventory)).digest('hex');
const request = {
  name: 'eidos-sentinel-lab', project: 'prj_5aL0COF1DONfPKI4fv25NitAdkOE', files,
  gitMetadata: { remoteUrl: 'https://github.com/bmparent/eidos', commitRef: environmentBranch, commitSha: sha, dirty: Boolean(git('status', '--porcelain')) },
  meta: { paidTemplatesGitSha: sha, paidTemplatesSourceBranch: git('branch', '--show-current'), paidTemplatesEnvironmentBranch: environmentBranch, paidTemplatesSourceSha256: bundleHash, manualOnly: String(manualOnly) },
  projectSettings: { rootDirectory: app.slice(0, -1) },
  ...(manualOnly ? { env: { EIDOS_SOURCE_COMMIT: sha, EIDOS_SOURCE_BUNDLE_HASH: bundleHash } } : {}),
};
const output = path.resolve(outputArg.slice('--out='.length));
if (output.startsWith(path.resolve(root, app, 'public') + path.sep)) throw Error('Private request cannot be written into public/.');
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(request), { mode: 0o600 });
await mkdir('artifacts/stripe-completion-20261014', { recursive: true });
await writeFile('artifacts/stripe-completion-20261014/bundle-inventory.json', JSON.stringify({ timestampUtc: new Date().toISOString(), backendSha: sha, bundleHash, archiveCount: count, files: inventory }, null, 2) + '\n');
console.log(JSON.stringify({ backendSha: sha, bundleHash, archiveCount: count, fileCount: files.length, requestBytes: Buffer.byteLength(JSON.stringify(request)), manualOnly, secretValuesRead: false }));
