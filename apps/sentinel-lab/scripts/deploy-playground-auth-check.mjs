/** Separate protected Preview. Does not modify branch secrets, aliases or the manual preview. */
import { readFile, writeFile, mkdir, lstat } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { vercelApi } from './vercel-preview-client.mjs';

const cli = process.argv.find(s => s.startsWith('--cli='))?.slice(6);
const countProbe = process.argv.includes('--count-probe');
const configuration = process.argv.find(s => s.startsWith('--configuration='))?.slice('--configuration='.length);
const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
const branch = 'codex/playground-unified-20261014', project = 'prj_5aL0COF1DONfPKI4fv25NitAdkOE', team = 'team_TEsnMvKU6SdbYjVhws6Ttp8H';
const artifactRoot = path.resolve('artifacts/playground-authenticated-20261015');
const out = path.resolve(process.argv.find(s => s.startsWith('--out='))?.slice(6) || artifactRoot);
if (out !== artifactRoot && !out.startsWith(artifactRoot + path.sep)) throw Error('Keep diagnostic receipts inside the dedicated artifact folder.');
if (!cli || git('branch', '--show-current') !== branch || git('diff', 'HEAD', '--', 'apps/sentinel-lab')) throw Error('Use committed isolated application source and authenticated CLI.');
const files = [], inventory = [];
for (const file of git('ls-files', '--', 'apps/sentinel-lab').split('\n').sort()) {
  if (!file || /(^|\/)(\.env[^/]*|\.vercel|private|tests|node_modules|artifacts|docs)(\/|$)/.test(file) || /\.(log|tsbuildinfo)$/.test(file)) continue;
  const stat = await lstat(file);
  if (!stat.isFile() || stat.isSymbolicLink()) throw Error('Unsafe source file.');
  const data = await readFile(file);
  files.push({ file, encoding: 'base64', data: data.toString('base64') });
  inventory.push({ file, bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') });
}
if (!files.some(f => f.file.endsWith('/playground-provider-check/route.ts'))) throw Error('Missing reviewed diagnostic.');
const sourceCommit = git('rev-parse', 'HEAD'), bundleHash = createHash('sha256').update(JSON.stringify(inventory)).digest('hex');
const env = { ...JSON.parse(await readFile('apps/sentinel-lab/.vercel/playground-manual.private.json', 'utf8')).env,
  EIDOS_SOURCE_COMMIT: sourceCommit, EIDOS_SOURCE_BUNDLE_HASH: bundleHash, EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS: 'test-auth',
  EIDOS_PLAYGROUND_AI_ENABLED: 'false', EIDOS_PLAYGROUND_INPUT_COUNT_PRICING: '' };
if (countProbe) {
  if (!configuration) throw Error('Supply the existing ignored approved TEST database configuration.');
  const input = JSON.parse(await readFile(configuration, 'utf8')).env;
  if (input.EIDOS_DATABASE_BINDING_PREFIX !== 'EIDOS_TEMPLATES_TEST' || !new URL(input.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation') || !input.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN)
    throw Error('Only the approved isolated TEST database may persist the count hold.');
  for (const key of ['EIDOS_DATABASE_BINDING_PREFIX','EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL','EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN']) env[key] = input[key];
  env.EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS = 'test-count-one';
}
// Inherit the explicitly approved branch Secret inside Vercel. Never read its value locally.
delete env.EIDOS_PLAYGROUND_OPENAI_API_KEY;
if (['EIDOS_ACCOUNTS_ENABLED','EIDOS_AI_ENABLED','EIDOS_SHOP_ENABLED','EIDOS_TEMPLATE_TEST_ENABLED','EIDOS_TEMPLATE_LIVE_ENABLED'].some(k => env[k] !== 'false') ||
    ['EIDOS_DATABASE_URL','EIDOS_DATABASE_AUTH_TOKEN'].some(k => env[k] !== '') ||
    (!countProbe && ['EIDOS_DATABASE_BINDING_PREFIX','EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN'].some(k => env[k] !== ''))) throw Error('Diagnostic must disable all customer/provider generation and exclude fallback databases.');
const receipt = { timestampUtc: new Date().toISOString(), branch, project, team, sourceCommit, bundleHash, files: inventory,
  dedicatedKey: 'Inherited approved Preview branch Secret; never read locally', accounts: false, inference: false, tokenCounting: countProbe ? 'one operator probe; unknown billing holds all $5' : false, database: countProbe ? 'approved TEST validation binding' : false, aliasChanged: false };
await mkdir(out, { recursive: true });
await writeFile(out + (countProbe ? '/count-source.json' : '/auth-source.json'), JSON.stringify(receipt, null, 2) + '\n');
if (process.argv.includes('--deploy')) {
  const result = vercelApi(cli, '/v13/deployments?forceNew=1&teamId=' + team, 'POST', {
    name: 'eidos-sentinel-lab', project, files, env, projectSettings: { rootDirectory: 'apps/sentinel-lab' },
    gitMetadata: { remoteUrl: 'https://github.com/bmparent/eidos', commitRef: branch, commitSha: sourceCommit, dirty: false },
    meta: { playgroundDiagnostic: countProbe ? 'count-one' : 'auth-only', playgroundGitSha: sourceCommit, playgroundBundleHash: bundleHash, paidTemplatesEnvironmentBranch: branch },
  });
  if (result.target === 'production') throw Error('Unexpected deployment target. Inspect immediately.');
  await writeFile(out + (countProbe ? '/count-deployment.json' : '/auth-deployment.json'), JSON.stringify({ timestampUtc: new Date().toISOString(), sourceCommit, bundleHash, id: result.id, url: 'https://' + result.url, state: result.readyState, target: result.target || 'preview' }, null, 2) + '\n');
  console.log(JSON.stringify({ id: result.id, url: 'https://' + result.url, sourceCommit, bundleHash, state: result.readyState, target: result.target || 'preview' }));
} else console.log(JSON.stringify({ sourceCommit, bundleHash, fileCount: files.length, dryRun: true }));
