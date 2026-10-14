/** New branch only. Existing isolated TEST relay/database remain in the ignored configuration input. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { vercelApi } from './vercel-preview-client.mjs';
const args = Object.fromEntries(process.argv.slice(2).filter(s => s.includes('=')).map(s => { const i = s.indexOf('='); return [s.slice(2, i), s.slice(i + 1)]; }));
const branch = execFileSync('git', ['branch', '--show-current'], { encoding: 'utf8' }).trim();
if (branch !== 'codex/playground-unified-20261014') throw Error('Run only from the isolated unified Playground branch.');
const manualOnly = process.argv.includes('--manual-only');
if ((!args.configuration && !manualOnly) || !args.cli) throw Error('Supply the existing ignored TEST configuration or --manual-only, and authenticated CLI entry.');
const input = args.configuration ? JSON.parse(await readFile(args.configuration, 'utf8')).env || {} : manualOnly ? await readFile('apps/sentinel-lab/.vercel/playground-manual.private.json', 'utf8').then(raw => JSON.parse(raw).env || {}).catch(error => { if (error.code !== 'ENOENT') throw error; return {}; }) : {};
if (manualOnly && !input.EIDOS_PLATFORM_TOKEN) input.EIDOS_PLATFORM_TOKEN = randomBytes(32).toString('hex');
if (!manualOnly && (input.EIDOS_DATABASE_BINDING_PREFIX !== 'EIDOS_TEMPLATES_TEST' || !new URL(input.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation') || !input.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN || (input.EIDOS_PLATFORM_TOKEN?.length || 0) < 32))
  throw Error('The existing isolated validation database and relay are required.');
const origin = 'https://eidosworks-playground-20261014.pages.dev';
const env = Object.fromEntries(['EIDOS_DATABASE_BINDING_PREFIX', 'EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL', 'EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN', 'EIDOS_PLATFORM_TOKEN', 'VERCEL_AUTOMATION_BYPASS_SECRET'].filter(key => input[key]).map(key => [key, input[key]]));
Object.assign(env, { PUBLIC_SITE_URL: origin, EIDOS_PREVIEW_ORIGINS: origin,
  EIDOS_SHOP_ENABLED: 'false', EIDOS_TEMPLATE_TEST_ENABLED: 'false', EIDOS_TEMPLATE_LIVE_ENABLED: 'false',
  EIDOS_ACCOUNTS_ENABLED: 'false', EIDOS_AI_ENABLED: 'false', EIDOS_NEWSLETTER_ENABLED: 'false', EIDOS_PROACTIVE_ENABLED: 'false', EIDOS_COMMUNITY_STUDIO_ENABLED: 'false',
  OPENAI_API_KEY: '', EIDOS_PLAYGROUND_AI_MODE: '', EIDOS_PLAYGROUND_DELIVERY_MODE: '', EIDOS_PLAYGROUND_STRIPE_KEY: '', EIDOS_PLAYGROUND_WEBHOOK_SECRET: '', RESEND_API_KEY: '', STRIPE_SECRET_KEY: '', EIDOS_KIT_WEBHOOK_SECRET: '', EIDOS_TEMPLATE_SOURCE_REVISION: '',
  EIDOS_DATABASE_URL: '', EIDOS_DATABASE_AUTH_TOKEN: '', EIDOS_OPERATOR_TOKEN: '', EIDOS_ADMIN_TOKEN: '', EIDOS_MAINTENANCE_TOKEN: '', EIDOS_RUNNER_TOKEN: '', EIDOS_TEST_ACCESS_GRANTS: '[]',
  EIDOS_RATE_SECRET: '', TURNSTILE_SECRET_KEY: '', KAGGLE_API_TOKEN: '', VERCEL_TOKEN: '', EIDOS_RUNNER_URL: '', EIDOS_ENABLE_FULL_CAPACITY: '0' });
if (manualOnly) Object.assign(env, { EIDOS_DATABASE_BINDING_PREFIX: '', EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL: '', EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN: '' });
const receipt = { branch, target: 'preview', project: 'prj_5aL0COF1DONfPKI4fv25NitAdkOE', team: 'team_TEsnMvKU6SdbYjVhws6Ttp8H', siteOrigin: origin,
  manualOnly, databaseBindingPrefix: env.EIDOS_DATABASE_BINDING_PREFIX || null, databaseHost: manualOnly ? null : new URL(env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname,
  features: 'Accounts, AI, commerce, mail, proactive and community studios disabled', configured: false, variableNames: Object.keys(env), timestampUtc: new Date().toISOString() };
if (process.argv.includes('--apply')) {
  const inherited = vercelApi(args.cli, '/v10/projects/' + receipt.project + '/env?teamId=' + receipt.team);
  const rows = inherited.envs || inherited;
  if (!Array.isArray(rows)) throw Error('Cannot inspect Preview inheritance.');
  receipt.inheritedPreviewVariableNames = rows.filter(row => row.target?.includes('preview') && !row.gitBranch).map(row => row.key).sort();
  // The manual milestone uses no inherited provider/database/signing credentials.
  // Deployment protection is the sole approved existing server-side exception.
  if (manualOnly) for (const key of receipt.inheritedPreviewVariableNames) {
    if (/(DATABASE|TURSO|_TOKEN$|_KEY$|_SECRET$)/.test(key) && !['EIDOS_PLATFORM_TOKEN', 'VERCEL_AUTOMATION_BYPASS_SECRET'].includes(key)) env[key] = '';
  }
  receipt.variableNames = Object.keys(env).sort();
  // Every database selection path supported by platformDatabase is explicitly overridden above.
  if (manualOnly && ['EIDOS_DATABASE_URL', 'EIDOS_DATABASE_AUTH_TOKEN', 'EIDOS_DATABASE_BINDING_PREFIX', 'EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL', 'EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN'].some(key => env[key] !== '')) throw Error('Manual-only Preview could inherit a database.');
  if (manualOnly) {
    await mkdir('apps/sentinel-lab/.vercel', { recursive: true });
    await writeFile('apps/sentinel-lab/.vercel/playground-manual.private.json', JSON.stringify({ env }), { mode: 0o600 });
  }
  for (const [key, value] of Object.entries(env)) {
    const result = vercelApi(args.cli, '/v10/projects/' + receipt.project + '/env?upsert=true&teamId=' + receipt.team, 'POST', { key, value, type: /SECRET|TOKEN|KEY/.test(key) ? 'sensitive' : 'encrypted', target: ['preview'], gitBranch: branch });
    if (!result.created || result.failed?.length) throw Error('New branch Preview configuration failed. Existing branches and Production were not retargeted.');
  }
  const after = vercelApi(args.cli, '/v10/projects/' + receipt.project + '/env?teamId=' + receipt.team);
  const scoped = (after.envs || after).filter(row => row.target?.includes('preview') && row.gitBranch === branch);
  if (Object.keys(env).some(key => !scoped.some(row => row.key === key))) throw Error('Preview branch override inventory is incomplete; do not deploy.');
  receipt.scopedOverrideNames = scoped.map(row => row.key).sort();
  receipt.inheritanceVerified = true;
  receipt.configured = true;
  if (manualOnly) {
    execFileSync('gh', ['secret', 'set', 'EIDOS_PLAYGROUND_PLATFORM_TOKEN', '--repo', 'bmparent/brent-parent-intelligence-studio'], { input: env.EIDOS_PLATFORM_TOKEN, stdio: ['pipe', 'pipe', 'pipe'] });
    receipt.frontendSecret = 'EIDOS_PLAYGROUND_PLATFORM_TOKEN';
  }
}
await mkdir('artifacts/playground-unified-backend-20261014', { recursive: true });
await writeFile('artifacts/playground-unified-backend-20261014/preview-environment.json', JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
