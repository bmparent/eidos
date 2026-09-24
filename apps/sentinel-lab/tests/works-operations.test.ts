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
    await client.execute('CREATE TABLE eidos_email_members(id TEXT PRIMARY KEY,email TEXT,username TEXT UNIQUE COLLATE NOCASE,kind TEXT,created_at TEXT,disabled INTEGER)');
    await client.execute("INSERT INTO eidos_email_members VALUES('member-a','a@example.com','alpha','person','2026-09-24',0)");
    await client.execute("INSERT INTO eidos_email_members VALUES('member-b','b@example.com','beta','person','2026-09-24',0)");
    await client.execute('CREATE TABLE eidos_member_sessions(token_hash TEXT PRIMARY KEY,member_id TEXT,expires INTEGER)');
    await client.execute("INSERT INTO eidos_member_sessions VALUES('session-a','member-a',9999999999999)");
    await client.execute("INSERT INTO eidos_member_sessions VALUES('session-b','member-b',9999999999999)");
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
    assert.equal(result.data.accounts,2);
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
    const noteRequest={command:'work_note',id:workId,note:'Follow up on the provider gate',reason:'Acceptance history',idempotencyKey:'88888888-8888-4888-8888-888888888888'};
    assert.equal((await operations(action(noteRequest),source,adaptDatabase(client))).status,201);
    assert.equal((await operations(action(noteRequest),source,adaptDatabase(client))).status,200);
    const reopened=await operations(action({command:'work_notes',id:workId}),source,adaptDatabase(client));
    assert.equal((await reopened.json()).data[0].body,noteRequest.note);
    assert.equal((await client.execute('SELECT COUNT(*) AS n FROM eidos_ops_work_notes')).rows[0].n,1);
    const stale=await operations(action({command:'work_update',id:workId,status:'done',version:0,reason:'Stale attempt',idempotencyKey:'22222222-2222-4222-8222-222222222222'}),source,adaptDatabase(client));
    assert.equal(stale.status,409);
    assert.equal((await client.execute("SELECT COUNT(*) AS n FROM eidos_ops_audit WHERE outcome='denied_stale_or_invalid'")).rows[0].n,1);
    const memberAction=(actionName:string,extra:Record<string,unknown>={},keyNumber=3)=>action({command:'account_action',memberId:'member-a',action:actionName,confirmation:'member-a',expectedUsername:extra.expectedUsername||'alpha',expectedDisabled:extra.expectedDisabled??0,reason:'Controlled account test',idempotencyKey:`${String(keyNumber).repeat(8)}-${String(keyNumber).repeat(4)}-4${String(keyNumber).repeat(3)}-8${String(keyNumber).repeat(3)}-${String(keyNumber).repeat(12)}`,...extra});
    assert.equal((await operations(memberAction('suspend'),source,adaptDatabase(client))).status,200);
    assert.equal((await client.execute("SELECT disabled FROM eidos_email_members WHERE id='member-a'")).rows[0].disabled,1);
    assert.equal((await client.execute("SELECT disabled FROM eidos_email_members WHERE id='member-b'")).rows[0].disabled,0);
    assert.equal((await operations(memberAction('restore',{expectedDisabled:1},4),source,adaptDatabase(client))).status,200);
    assert.equal((await operations(memberAction('username',{username:'beta'},5),source,adaptDatabase(client))).status,409);
    assert.equal((await operations(memberAction('username',{username:'gamma'},6),source,adaptDatabase(client))).status,200);
    assert.equal((await client.execute("SELECT username FROM eidos_email_members WHERE id='member-b'")).rows[0].username,'beta');
    assert.equal((await operations(memberAction('revoke_sessions',{expectedUsername:'gamma'},7),source,adaptDatabase(client))).status,200);
    assert.equal((await client.execute("SELECT COUNT(*) AS n FROM eidos_member_sessions WHERE member_id='member-a'")).rows[0].n,0);
    assert.equal((await client.execute("SELECT COUNT(*) AS n FROM eidos_member_sessions WHERE member_id='member-b'")).rows[0].n,1);
  } finally {globalThis.fetch=originalFetch;client.close();}
});
