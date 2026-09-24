import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { operations } from '../lib/works/operations';
import {generateKeyPair,exportJWK,SignJWT} from 'jose';

test('owner console denies missing and forged Access assertions before reading customer data', async () => {
  const client=createClient({url:'file::memory:'});
  try {
    const database=adaptDatabase(client);
    const env={EIDOS_OPS_ACCESS_TEAM:'example-team',EIDOS_OPS_ACCESS_AUD:'app-aud',EIDOS_OPS_OWNER_SUB:'stable-owner-id',EIDOS_OPS_OWNER_EMAIL:'owner@example.com'};
    const noToken=await operations(new Request('https://owner.example.com/api/operations/console'),env,database);
    assert.equal(noToken.status,403);
    const forged=await operations(new Request('https://owner.example.com/api/operations/console',{headers:{'cf-access-authenticated-user-email':'owner@example.com','cf-access-jwt-assertion':'forged'}}),env,database);
    assert.equal(forged.status,403);
  } finally { client.close(); }
});

test('additive operations migration retains existing member and payment records', async () => {
  const client=createClient({url:'file::memory:'});
  try {
    await client.execute('CREATE TABLE eidos_email_members(id TEXT PRIMARY KEY, email TEXT, username TEXT)');
    await client.execute("INSERT INTO eidos_email_members VALUES('member-1','test@example.com','tester')");
    await client.execute('CREATE TABLE eidos_orders(id TEXT PRIMARY KEY,status TEXT)');
    await client.execute("INSERT INTO eidos_orders VALUES('order-1','paid')");
    const {readFileSync}=await import('node:fs');
    const sql=readFileSync(new URL('../lib/works/vendor/migrations/0007_operations.sql',import.meta.url),'utf8');
    for(const statement of sql.split(';').map(x=>x.trim()).filter(Boolean)) await client.execute(statement);
    assert.equal((await client.execute('SELECT COUNT(*) AS n FROM eidos_email_members')).rows[0].n,1);
    assert.equal((await client.execute('SELECT COUNT(*) AS n FROM eidos_orders')).rows[0].n,1);
    assert.equal((await client.execute("SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name LIKE 'eidos_ops_%'")).rows[0].n,5);
  } finally { client.close(); }
});

test('a signed owner identity may read while a different subject is denied', async () => {
  const {publicKey,privateKey}=await generateKeyPair('RS256');
  const jwk=await exportJWK(publicKey);jwk.kid='test-key';jwk.alg='RS256';jwk.use='sig';
  const originalFetch=globalThis.fetch;
  globalThis.fetch=async (input,init)=>{
    if(String(input).includes('ops-test-team.cloudflareaccess.com/cdn-cgi/access/certs')) return new Response(JSON.stringify({keys:[jwk]}),{headers:{'content-type':'application/json'}});
    return originalFetch(input,init);
  };
  const source={EIDOS_OPS_ACCESS_TEAM:'ops-test-team',EIDOS_OPS_ACCESS_AUD:'ops-audience',EIDOS_OPS_OWNER_SUB:'owner-123',EIDOS_OPS_OWNER_EMAIL:'owner@example.com',EIDOS_SOURCE_REVISION:'test-revision',EIDOS_OPS_ORIGIN:'https://owner.example.com'};
  const client=createClient({url:'file::memory:'});
  try {
    await client.execute('CREATE TABLE eidos_email_members(id TEXT PRIMARY KEY)');
    await client.execute('CREATE TABLE eidos_orders(id TEXT,status TEXT)');
    await client.execute('CREATE TABLE eidos_agents(id TEXT,revoked INTEGER)');
    await client.execute('CREATE TABLE eidos_outcomes(day TEXT,feature TEXT,outcome TEXT,count INTEGER,updated TEXT)');
    const {readFileSync}=await import('node:fs');
    for(const statement of readFileSync(new URL('../lib/works/vendor/migrations/0007_operations.sql',import.meta.url),'utf8').split(';').map(x=>x.trim()).filter(Boolean)) await client.execute(statement);
    const sign=(sub:string)=>new SignJWT({email:'owner@example.com'}).setProtectedHeader({alg:'RS256',kid:'test-key'}).setIssuer('https://ops-test-team.cloudflareaccess.com').setAudience('ops-audience').setSubject(sub).setIssuedAt().setExpirationTime('5m').sign(privateKey);
    const denied=await operations(new Request('https://owner.example.com/api/operations/console',{headers:{'cf-access-jwt-assertion':await sign('other-123')}}),source,adaptDatabase(client));
    assert.equal(denied.status,403);
    const allowed=await operations(new Request('https://owner.example.com/api/operations/console',{headers:{'cf-access-jwt-assertion':await sign('owner-123')}}),source,adaptDatabase(client));
    assert.equal(allowed.status,200);
    const result=await allowed.json();
    assert.equal(result.data.sourceRevision,'test-revision');
    assert.equal(result.data.accounts,0);
    const ownerJwt=await sign('owner-123');
    const key='11111111-1111-4111-8111-111111111111';
    const input={command:'work_create',title:'Check isolated preview',detail:'Controlled test item',severity:'normal',reason:'Acceptance check',idempotencyKey:key};
    const action=(body:unknown,csrf=true)=>new Request('https://owner.example.com/api/operations/console',{method:'POST',headers:{'cf-access-jwt-assertion':ownerJwt,'content-type':'application/json',...(csrf?{'origin':'https://owner.example.com','x-ops-csrf':'1'}:{})},body:JSON.stringify(body)});
    assert.equal((await operations(action(input,false),source,adaptDatabase(client))).status,403);
    const created=await operations(action(input),source,adaptDatabase(client));
    assert.equal(created.status,201);
    const workId=(await created.json()).data.id;
    const duplicate=await operations(action(input),source,adaptDatabase(client));
    assert.equal((await duplicate.json()).replayed,true);
    assert.equal((await client.execute('SELECT COUNT(*) AS n FROM eidos_ops_work')).rows[0].n,1);
    const stale=await operations(action({command:'work_update',id:workId,status:'done',version:0,reason:'Stale attempt',idempotencyKey:'22222222-2222-4222-8222-222222222222'}),source,adaptDatabase(client));
    assert.equal(stale.status,409);
    assert.equal((await client.execute("SELECT COUNT(*) AS n FROM eidos_ops_audit WHERE outcome='denied_stale_or_invalid'")).rows[0].n,1);
  } finally {globalThis.fetch=originalFetch;client.close();}
});
