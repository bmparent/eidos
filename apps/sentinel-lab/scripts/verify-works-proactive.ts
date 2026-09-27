import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { onRequestPost } from '../lib/works/vendor/functions/api/community/maintenance';

// Deliberately uses only the isolated validation database and synthetic timestamps.
async function main() {
  assert.equal(process.env.PUBLIC_SITE_URL, 'https://release-validation-20260905.eidosworks.pages.dev');
  assert.ok(process.env.EIDOS_DATABASE_URL?.includes('eidos-works-validation'));
  const client = createClient({ url: process.env.EIDOS_DATABASE_URL!, authToken: process.env.EIDOS_DATABASE_AUTH_TOKEN });
  const db = adaptDatabase(client);
  const ids = Array.from({length: 15}, () => crypto.randomUUID());
  const old = new Date(Date.now() - 2 * 86400000).toISOString();
  const now = new Date().toISOString();
  try {
    for (let i = 0; i < ids.length; i++) {
      await client.execute({sql: "INSERT INTO eidos_threads(id,title,body,category,author,author_type,status,allow_assistant,created_at,published_at) VALUES(?,?,?,'build','Release QA',?,'published',?,?,?)",
        args: [ids[i], 'TEST proactive cinematic starter ' + i, 'TEST synthetic eligibility fixture for published studio kit knowledge.', i === 13 ? 'agent' : 'guest', i === 12 ? 0 : 1, old, i === 11 ? now : old]});
    }
    await client.execute({sql: "INSERT INTO eidos_replies(id,thread_id,body,author,author_type,status,created_at) VALUES(?,?,'TEST existing answer','Release QA','guest','published',?)",args:[crypto.randomUUID(),ids[14],now]});
    const quota = async () => Number((await client.execute({sql:"SELECT used FROM eidos_quotas WHERE bucket='public-eidos-suggestions' AND period=?",args:[Math.floor(Date.now()/86400000)]})).rows[0]?.used || 0);
    const before = await quota();
    assert.ok(before < 10, 'Validation daily suggestion budget already exhausted');
    const context = { request:new Request(process.env.PUBLIC_SITE_URL + '/api/community/maintenance',{method:'POST',headers:{authorization:'Bearer '+process.env.EIDOS_MAINTENANCE_TOKEN}}),env:{...process.env,EIDOS_DB:db,EIDOS_PROACTIVE_ENABLED:'true'} };
    const first = await onRequestPost(context); assert.equal(first.status,200);
    const result = await first.json() as {suggestions:number};
    assert.equal(result.suggestions,Math.min(10,10-before));
    assert.equal(await quota(),10);
    const second = await onRequestPost(context); assert.equal(second.status,200);
    assert.equal((await second.json() as {suggestions:number}).suggestions,0);
    for (const id of ids.slice(11)) {
      const r = await client.execute({sql:"SELECT COUNT(*) AS n FROM eidos_replies WHERE thread_id=? AND author_type='eidos'",args:[id]});
      assert.equal(Number(r.rows[0].n),0);
    }
    const artifact={timestamp_utc:now,execution:'actual maintenance handler with remote validation libSQL; synthetic aged fixtures',previousSuggestions:before,newSuggestions:result.suggestions,dailyTotal:10,repeatSuggestions:0,youngOptOutAgentAndAnsweredExcluded:true,cleanup:'fixture rows removed in finally'};
    fs.writeFileSync('artifacts/works-release-20260905/proactive-storage-smoke.json',JSON.stringify(artifact,null,2)+'\n');
    console.log(JSON.stringify(artifact));
  } finally {
    for(const id of ids){await client.execute({sql:'DELETE FROM eidos_replies WHERE thread_id=?',args:[id]});await client.execute({sql:'DELETE FROM eidos_threads WHERE id=?',args:[id]});}
    client.close();
  }
}
await main();
