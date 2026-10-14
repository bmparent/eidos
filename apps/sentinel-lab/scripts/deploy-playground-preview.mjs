/** Explicit manual-only deployment. Never enables commerce, AI, accounts or database persistence. */
import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { vercelApi } from './vercel-preview-client.mjs';
const options = Object.fromEntries(process.argv.slice(2).filter(value => value.startsWith('--') && value.includes('=')).map(value => { const i = value.indexOf('='); return [value.slice(2, i), value.slice(i + 1)]; }));
const branch = 'codex/playground-unified-20261014', project = 'prj_5aL0COF1DONfPKI4fv25NitAdkOE', team = 'team_TEsnMvKU6SdbYjVhws6Ttp8H';
const receiptPath = 'artifacts/playground-unified-backend-20261014/backend-preview.json';
const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
if (!options.cli || git('branch', '--show-current') !== branch) throw Error('Use the authenticated CLI from the isolated Playground branch.');
let receipt;
if (process.argv.includes('--inspect')) {
  receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
  if (receipt.branch !== branch || !receipt.deployment?.id) throw Error('No isolated deployment receipt to inspect.');
  const result = vercelApi(options.cli, '/v13/deployments/' + receipt.deployment.id + '?teamId=' + team);
  if (result.target === 'production' || result.projectId !== project) throw Error('Unexpected deployment identity.');
  if (result.id !== receipt.deployment.id || result.meta?.paidTemplatesGitSha !== receipt.backendSha ||
      result.meta?.paidTemplatesSourceSha256 !== receipt.bundleHash || result.meta?.manualOnly !== 'true' ||
      result.meta?.paidTemplatesEnvironmentBranch !== branch) throw Error('Deployment source metadata does not match the frozen manual-only candidate.');
  receipt.deployment = { id: result.id, url: 'https://' + result.url, state: result.readyState, target: result.target || 'preview' };
  receipt.sourceMetadataVerified = true;
  receipt.inspectedAt = new Date().toISOString();
} else {
  if (!options.bundle) throw Error('Supply the private bundle request generated from clean reviewed source.');
  const configuration = JSON.parse(await readFile('artifacts/playground-unified-backend-20261014/preview-environment.json', 'utf8'));
  if (!configuration.configured || !configuration.inheritanceVerified || !configuration.manualOnly || configuration.branch !== branch || configuration.databaseHost !== null)
    throw Error('Complete the scoped, database-free Preview configuration before deployment.');
  const payload = JSON.parse(await readFile(options.bundle, 'utf8'));
  const head = git('rev-parse', 'HEAD');
  if (payload.project !== project || payload.target || payload.gitMetadata?.commitRef !== branch || payload.gitMetadata?.commitSha !== head || payload.gitMetadata?.dirty || payload.meta?.manualOnly !== 'true' || git('diff', 'HEAD', '--', 'apps/sentinel-lab'))
    throw Error('Bundle must match the unchanged committed manual-only source and Preview target.');
  const inventory = [];
  for (const file of payload.files) {
    if (!file.file.startsWith('apps/sentinel-lab/') || file.file.split(/[\\/]/).includes('..') || /(^|\/)(\.env[^/]*|\.vercel|private\/templates)(\/|$)/.test(file.file) || file.encoding !== 'base64') throw Error('Unsafe manual-only bundle path.');
    const data = Buffer.from(file.data, 'base64');
    if (!data.equals(await readFile(file.file))) throw Error('Bundle source changed since freeze.');
    inventory.push({ file: file.file, bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') });
  }
  const bundleHash = createHash('sha256').update(JSON.stringify(inventory)).digest('hex');
  if (payload.env?.EIDOS_SOURCE_COMMIT !== head || payload.env?.EIDOS_SOURCE_BUNDLE_HASH !== bundleHash || payload.meta?.paidTemplatesSourceSha256 !== bundleHash) throw Error('Runtime/source bundle identities do not match.');
  // Repeat all isolated overrides on this deployment itself, so branch-selection
  // changes or inherited Preview defaults cannot activate a database/provider.
  const isolated = JSON.parse(await readFile('apps/sentinel-lab/.vercel/playground-manual.private.json', 'utf8')).env;
  if (!isolated || isolated.PUBLIC_SITE_URL !== 'https://eidosworks-playground-20261014.pages.dev' || (isolated.EIDOS_PLATFORM_TOKEN?.length || 0) < 32 ||
      ['EIDOS_SHOP_ENABLED', 'EIDOS_TEMPLATE_TEST_ENABLED', 'EIDOS_TEMPLATE_LIVE_ENABLED', 'EIDOS_ACCOUNTS_ENABLED', 'EIDOS_AI_ENABLED', 'EIDOS_NEWSLETTER_ENABLED', 'EIDOS_PROACTIVE_ENABLED', 'EIDOS_COMMUNITY_STUDIO_ENABLED'].some(key => isolated[key] !== 'false') ||
      ['EIDOS_DATABASE_URL', 'EIDOS_DATABASE_AUTH_TOKEN', 'EIDOS_DATABASE_BINDING_PREFIX', 'EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL', 'EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN', 'OPENAI_API_KEY', 'STRIPE_SECRET_KEY', 'RESEND_API_KEY', 'EIDOS_PLAYGROUND_AI_MODE', 'EIDOS_PLAYGROUND_DELIVERY_MODE'].some(key => isolated[key] !== '') ||
      configuration.variableNames.some(key => !(key in isolated))) throw Error('Isolated per-deployment environment is incomplete.');
  Object.assign(payload.env, isolated);
  receipt = { branch, backendSha: head, bundleHash, manualOnly: true, database: 'none', privateArchiveCount: 0,
    privateArchiveState: 'Intentionally omitted for disabled manual editor preview. Original complete TEST archive bundle retained unchanged.',
    features: 'Accounts, AI, commerce, mail, community and proactive disabled', perDeploymentOverrides: true, createdAt: new Date().toISOString(), deployed: false };
  if (process.argv.includes('--deploy')) {
    const result = vercelApi(options.cli, '/v13/deployments?forceNew=1&teamId=' + team, 'POST', payload);
    if (result.target === 'production') throw Error('Unexpected Production deployment; inspect provider immediately.');
    receipt.deployed = true;
    receipt.deployment = { id: result.id, url: 'https://' + result.url, state: result.readyState || result.status, target: result.target || 'preview' };
  }
}
await writeFile(receiptPath, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
