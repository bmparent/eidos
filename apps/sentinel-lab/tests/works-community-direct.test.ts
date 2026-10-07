import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient } from '@libsql/client';
import { readFileSync } from 'node:fs';
import { adaptDatabase } from '../lib/works/database';
import { dispatchWorks } from '../lib/works/dispatch';
import { hash } from '../lib/works/vendor/functions/_shared/platform/core';

test('Authenticated relay publishes human and agent conversations immediately on real libSQL, with removal and revocation', async () => {
  const client = createClient({url: ':memory:'});
  for (const name of ['0001_eidos_platform.sql','0002_members.sql'])
    await client.executeMultiple(readFileSync('lib/works/vendor/migrations/'+name,'utf8'));
  const site = 'https://eidos-works.com';
  const source = {EIDOS_PLATFORM_TOKEN:'test-relay-credential-with-at-least-32-characters',PUBLIC_SITE_URL:site};
  const env = {EIDOS_DB:adaptDatabase(client),EIDOS_RATE_SECRET:'test-rate-credential-with-at-least-32-characters',
    EIDOS_ADMIN_TOKEN:'test-admin-credential-with-at-least-32-characters',TURNSTILE_SITE_KEY:'test-site',TURNSTILE_SECRET_KEY:'test-secret',
    EIDOS_ACCOUNTS_ENABLED:'true',RESEND_API_KEY:'test-no-mail',EIDOS_MAIL_FROM:'QA <qa@example.test>'};
  const memberId=crypto.randomUUID(),session='a'.repeat(64),now=new Date().toISOString();
  await client.execute({sql:"INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,'person',?,?)",args:[memberId,'qa@example.test','qa_person',now,now]});
  await client.execute({sql:'INSERT INTO eidos_member_sessions(member_id,token_hash,expires) VALUES(?,?,?)',args:[memberId,await hash(session),Math.floor(Date.now()/1000)+3600]});
  const req = (path:string,body?:unknown,headers:Record<string,string>={}) => new Request('https://backend.vercel.app/api/works/v1'+path,{
    method:body===undefined?'GET':'POST',headers:{'x-eidos-platform-token':source.EIDOS_PLATFORM_TOKEN,'x-eidos-site-origin':site,origin:site,'content-type':'application/json',...headers},
    ...(body===undefined?{}:{body:JSON.stringify(body)}),
  });
  const call = (path:string,body?:unknown,headers:Record<string,string>={}) => dispatchWorks(req(path,body,headers),source,env);
  const original=globalThis.fetch;
  globalThis.fetch=async()=>{throw Error('No CAPTCHA, mail or model calls for authenticated contributions');};
  try {
    const created=await call('/api/community/threads',{title:'A useful human question',body:'How can we simplify this layout?',author:'Forged name'},{cookie:'__Host-eidos_session='+session});
    assert.equal(created.status,201,await created.clone().text());
    const thread=await created.json();assert.equal(thread.state,'published');
    const current=await (await call('/api/community/threads?id='+thread.id)).json();
    assert.equal(current.thread.author,'@qa_person');assert.ok(current.thread.published_at);
    assert.equal('owner_id' in current.thread,false);
    assert.equal('allow_assistant' in current.thread,false);
    const registered=await (await call('/api/community/moderate',{action:'register-agent',name:'QA Agent',profileUrl:'https://example.com/operator'},{authorization:'Bearer '+env.EIDOS_ADMIN_TOKEN})).json();
    const agentHeaders={authorization:'Bearer '+registered.key};
    const response=await call('/api/community/agents',{threadId:thread.id,body:'Yes!'},agentHeaders);
    assert.equal(response.status,201);const reply=await response.json();assert.equal(reply.state,'published');
    const after=await (await call('/api/community/threads?id='+thread.id)).json();
    assert.equal(after.thread.reply_count,1);assert.equal(after.replies[0].author_type,'agent');
    assert.equal((await call('/api/community/moderate',{action:'unpublish',kind:'reply',id:reply.id})).status,401);
    await call('/api/community/moderate',{action:'unpublish',kind:'reply',id:reply.id},{authorization:'Bearer '+env.EIDOS_ADMIN_TOKEN});
    assert.equal((await (await call('/api/community/threads?id='+thread.id)).json()).thread.reply_count,0);
    await call('/api/community/moderate',{action:'revoke-agent',id:registered.id},{authorization:'Bearer '+env.EIDOS_ADMIN_TOKEN});
    assert.equal((await call('/api/community/agents',{threadId:thread.id,body:'More'},agentHeaders)).status,401);
    assert.equal((await dispatchWorks(req('/api/community/threads',undefined,{'x-eidos-site-origin':'https://untrusted.example'}),source,env)).status,403);
    const invalidSession=await call('/api/community/replies',{threadId:thread.id,body:'Hi',author:'Guest'},{cookie:'__Host-eidos_session='+'b'.repeat(64)});
    assert.equal(invalidSession.status,400);
  } finally {globalThis.fetch=original;client.close();}
});
