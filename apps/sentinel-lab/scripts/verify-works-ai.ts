import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { adaptDatabase } from '../lib/works/database';
import { dispatchWorks } from '../lib/works/dispatch';

// Explicit release smoke: one real provider call, local synthetic state only.
// Run from repository root: node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-ai.ts
assert.ok(process.env.OPENAI_API_KEY, 'OPENAI_API_KEY must be supplied securely');
const client = createClient({ url: ':memory:' });
await client.executeMultiple(await readFile(new URL('../lib/works/vendor/migrations/0001_eidos_platform.sql', import.meta.url), 'utf8'));
const database = adaptDatabase(client);
const secret = randomUUID() + randomUUID();
const source = { EIDOS_PLATFORM_TOKEN: secret, PUBLIC_SITE_URL: 'https://eidos-works.com' };
const env = {
  EIDOS_RUNTIME: 'sentinel' as const, EIDOS_DB: database, EIDOS_AI_ENABLED: 'true',
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  EIDOS_ASSISTANT_MODEL: 'gpt-4.1-nano-2025-04-14', EIDOS_AI_DAILY_TOKENS: '20000',
  EIDOS_RATE_SECRET: randomUUID() + randomUUID(),
};
const input = { question: 'What kinds of websites can Eidos Works help build?', enhanced: true, requestId: randomUUID() };
function request(payload: unknown) {
  return new Request('https://eidos-sentinel-lab.vercel.app/api/works/v1/api/assistant', {
    method: 'POST', headers: { 'x-eidos-platform-token': secret,
      'x-eidos-site-origin': source.PUBLIC_SITE_URL, origin: source.PUBLIC_SITE_URL,
      'content-type': 'application/json', 'x-eidos-client-ip': '192.0.2.29' },
    body: JSON.stringify(payload),
  });
}
const original = globalThis.fetch;
const evidence: Record<string, unknown> = { timestamp: new Date().toISOString(), model: env.EIDOS_ASSISTANT_MODEL,
  provider: 'real OpenAI Responses API', database: 'local in-memory libSQL; not deployed persistence',
  max_output_tokens: 320, retry_count: 0, calls: 0 };
globalThis.fetch = async (url, init) => {
  assert.equal(String(url), 'https://api.openai.com/v1/responses');
  evidence.calls = Number(evidence.calls) + 1;
  assert.equal(evidence.calls, 1, 'Smoke must not make multiple paid requests');
  const payload = JSON.parse(String(init?.body));
  assert.equal(payload.max_output_tokens, 320);
  assert.equal(payload.store, false);
  assert.equal(payload.tools, undefined);
  const response = await original(url, init);
  const result = await response.clone().json();
  evidence.provider_status = response.status;
  evidence.request_id = response.headers.get('x-request-id');
  evidence.usage = result.usage ?? null;
  evidence.error_code = result.error?.code ?? null;
  evidence.error_type = result.error?.type ?? null;
  return response;
};
try {
  const first = await (await dispatchWorks(request(input), source, env)).json();
  evidence.answer_mode = first.mode;
  evidence.answer_characters = first.answer?.length ?? 0;
  const duplicate = await (await dispatchWorks(request(input), source, env)).json();
  assert.deepEqual(duplicate, first);
  evidence.idempotency_passed = true;
  const disabled = await (await dispatchWorks(request({ ...input, requestId: randomUUID() }), source,
    { ...env, EIDOS_AI_DAILY_TOKENS: '0' })).json();
  assert.equal(disabled.mode, 'sources');
  evidence.zero_budget_fallback_passed = true;
  const noStorage = await (await dispatchWorks(request(input), source, { ...env, EIDOS_DB: undefined })).json();
  assert.equal(noStorage.mode, 'sources');
  evidence.missing_storage_fallback_passed = true;
  evidence.reservation = await database.prepare("SELECT used FROM eidos_quotas WHERE bucket='ai-global'").first();
  assert.equal(evidence.calls, 1);
  evidence.passed = first.mode === 'ai' && evidence.provider_status === 200;
} finally {
  globalThis.fetch = original;
  client.close();
  const output = new URL('../../../artifacts/works-release-20260905/', import.meta.url);
  await mkdir(output, { recursive: true });
  await writeFile(new URL('ai-provider-smoke.json', output), JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify(evidence));
}
if (!evidence.passed) process.exitCode = 1;
