/** Additive setup for the existing approved validation database; never creates paid/member rows. */
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { aiPolicyStatus, configureAiPolicy, ensureAiLedger, USD } from '../lib/works/vendor/functions/_shared/platform/playgroundAi';
import { providerTestRateCard } from '../lib/works/vendor/functions/_shared/platform/playgroundAiProvider';

const configuration = process.argv.find(s => s.startsWith('--configuration='))?.slice('--configuration='.length);
if (!configuration) throw Error('Supply the existing ignored TEST configuration path.');
const env = JSON.parse(await readFile(configuration, 'utf8')).env as Record<string, string>;
if (env.EIDOS_DATABASE_BINDING_PREFIX !== 'EIDOS_TEMPLATES_TEST' || !new URL(env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation') || !env.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN)
  throw Error('Refusing a database outside the approved TEST validation binding.');
const client = createClient({ url: env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL, authToken: env.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN });
const platform = { EIDOS_DB: adaptDatabase(client) };
const protectedTables = ['eidos_email_members','eidos_orders','eidos_template_orders','eidos_pg_projects','eidos_pg_orders'];
const counts = async () => Object.fromEntries(await Promise.all(protectedTables.map(async name => [name, Number((await client.execute('SELECT COUNT(*) count FROM ' + name)).rows[0].count)])));
const receipt: Record<string, unknown> = { timestampUtc: new Date().toISOString(), validationDatabase: true, mode: 'test',
  sourceCommit: execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(), providerCalls: 0, secretValuesPrinted: false,
  requestedBudgetMicros: 5 * USD, enabled: false, dryRun: !process.argv.includes('--apply') };
try {
  const before = await counts(); receipt.protectedCountsBefore = before;
  const existing = await client.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='eidos_pg_ai_policies'");
  if (process.argv.includes('--apply')) {
    await ensureAiLedger(platform);
    const policy = await platform.EIDOS_DB.prepare('SELECT mode FROM eidos_pg_ai_policies WHERE mode=?').bind('test').first();
    if (!policy) await configureAiPolicy(platform, 'test', providerTestRateCard('gpt-6.1-sol'), 5 * USD, false);
    receipt.policy = await aiPolicyStatus(platform, 'test');
    receipt.schemaSha256 = createHash('sha256').update(await readFile('apps/sentinel-lab/lib/works/vendor/migrations/0005_playground_ai.sql')).digest('hex');
  } else receipt.ledgerAlreadyExists = existing.rows.length > 0;
  const after = await counts(); receipt.protectedCountsAfter = after;
  if (JSON.stringify(before) !== JSON.stringify(after)) throw Error('Protected record counts changed; inspect before proceeding.');
  receipt.protectedRecordsPreserved = true; receipt.status = 'passed';
} catch {
  receipt.status = 'failed'; receipt.reason = 'TEST setup failed; private connection and database details suppressed.'; process.exitCode = 1;
} finally {
  client.close(); await mkdir('artifacts/playground-authenticated-20261015',{recursive:true});
  await writeFile('artifacts/playground-authenticated-20261015/test-ledger-'+(process.argv.includes('--apply')?'setup':'dry-run')+'.json',JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify(receipt));
}
