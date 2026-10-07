import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createClient } from '@libsql/client';
import { adaptDatabase, databaseConfiguration } from '../lib/works/database';
import { dispatchWorks, platformEnvironment } from '../lib/works/dispatch';
import { hash } from '../lib/works/vendor/functions/_shared/platform/core';

// Explicit isolated QA identities; never run against production or send mail.
const receipt: Record<string, unknown> = {marker:'EIDOS_COMMUNITY_DIRECT_ACCEPTANCE',time:new Date().toISOString(),source:process.env.VERCEL_GIT_COMMIT_SHA,status:'failed',identityMode:'synthetic QA sessions; real isolated libSQL; no signup/email proof'};
let client: ReturnType<typeof createClient> | undefined;
try {
  assert.equal(process.env.VERCEL_ENV,'preview');
  assert.equal(process.env.VERCEL_GIT_COMMIT_REF,'codex/community-direct-replies-20261007');
  assert.equal(process.env.EIDOS_DATABASE_BINDING_PREFIX,'EIDOS_PG_INTEGRATION');
  const binding=databaseConfiguration(process.env);assert.ok(binding.url&&binding.authToken);
  const url=new URL(binding.url);
  const identity=createHash('sha256').update(url.hostname+url.pathname).digest('hex');
  assert.equal(identity,'4882b7d6be40a19a97d5dabce311089dc72a984e5a2b3b3f64081216926bb353');
  receipt.databaseIdentityHash=identity;
  client=createClient({url:binding.url,authToken:binding.authToken});
  const live=platformEnvironment(process.env);assert.ok(live.EIDOS_ADMIN_TOKEN&&(live.EIDOS_ADMIN_TOKEN.length>=32));
  assert.ok(process.env.EIDOS_PLATFORM_TOKEN&&(process.env.EIDOS_PLATFORM_TOKEN.length>=32));
  // In-process fixture readiness only. Deployed account/provider flags stay off.
  const env={...live,EIDOS_DB:adaptDatabase(client),EIDOS_ACCOUNTS_ENABLED:'true',
    RESEND_API_KEY:'qa-fixture-no-mail',EIDOS_MAIL_FROM:'QA <qa@example.invalid>',TURNSTILE_SECRET_KEY:'qa-fixture-no-call',TURNSTILE_SITE_KEY:'qa-fixture-no-call'};
  const source: Record<string,string|undefined>={...process.env,PUBLIC_SITE_URL:'https://eidosworks-test-20260923.pages.dev'};
  const person=crypto.randomUUID(),session=crypto.randomUUID().replaceAll('-','')+crypto.randomUUID().replaceAll('-',''),now=new Date().toISOString();
  const username='qa_person_'+person.replaceAll('-','').slice(0,8);
  await client.execute({sql:"INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,'person',?,?)",args:[person,'qa-'+person+'@example.invalid',username,now,now]});
  await client.execute({sql:'INSERT INTO eidos_member_sessions(member_id,token_hash,expires) VALUES(?,?,?)',args:[person,await hash(session),Math.floor(Date.now()/1000)+3600]});
  const call=(path:string,body?:unknown,headers:Record<string,string>={})=>dispatchWorks(new Request('https://backend.vercel.app/api/works/v1'+path,{
    method:body===undefined?'GET':'POST',headers:{'x-eidos-platform-token':source.EIDOS_PLATFORM_TOKEN!,'x-eidos-site-origin':source.PUBLIC_SITE_URL!,origin:source.PUBLIC_SITE_URL!,'content-type':'application/json',...headers},...(body===undefined?{}:{body:JSON.stringify(body)}),
  }),source,env);
  const created=await call('/api/community/threads',{title:'Isolated QA: immediate community conversations',body:'Isolated preview QA contribution. This verifies immediate human posting and labeled agent replies; it is not a customer conversation.',category:'build'},{cookie:'__Host-eidos_session='+session});
  assert.equal(created.status,201);const thread=await created.json();assert.equal(thread.state,'published');
  const admin={authorization:'Bearer '+live.EIDOS_ADMIN_TOKEN};
  const registeredResponse=await call('/api/community/moderate',{action:'register-agent',name:'Isolated Community QA Agent',profileUrl:'https://eidos-works.com/community/agent-guide'},admin);
  assert.equal(registeredResponse.status,201);const agent=await registeredResponse.json();
  const response=await call('/api/community/agents',{threadId:thread.id,body:'Isolated QA agent reply: visible immediately without review.'},{authorization:'Bearer '+agent.key});
  assert.equal(response.status,201);const reply=await response.json();assert.equal(reply.state,'published');
  const detail=await (await call('/api/community/threads?id='+thread.id)).json();
  assert.equal(detail.thread.author,'@'+username);assert.equal(detail.thread.reply_count,1);assert.equal(detail.replies[0].id,reply.id);assert.equal(detail.replies[0].author_type,'agent');
  const page=await call('/community/thread/'+thread.id);assert.equal(page.status,200);assert.match(await page.text(),/Replies appear immediately/);
  const queue=await (await call('/api/community/moderate',undefined,admin)).json();assert.ok(!queue.threads.some((t:{id:string})=>t.id===thread.id));assert.ok(!queue.replies.some((r:{id:string})=>r.id===reply.id));
  assert.equal((await call('/api/community/moderate',{action:'unpublish',kind:'reply',id:reply.id})).status,401);
  assert.equal((await call('/api/community/moderate',{action:'unpublish',kind:'reply',id:reply.id},admin)).status,200);
  assert.equal((await (await call('/api/community/threads?id='+thread.id)).json()).thread.reply_count,0);
  assert.equal((await call('/api/community/moderate',{action:'publish',kind:'reply',id:reply.id},admin)).status,200);
  assert.equal((await call('/api/community/moderate',{action:'revoke-agent',id:agent.id},admin)).status,200);
  assert.equal((await call('/api/community/agents',{threadId:thread.id,body:'Should be denied.'},{authorization:'Bearer '+agent.key})).status,401);
  await client.execute({sql:'DELETE FROM eidos_member_sessions WHERE member_id=?',args:[person]});
  await client.execute({sql:'UPDATE eidos_email_members SET disabled=1 WHERE id=?',args:[person]});
  receipt.threadId=thread.id;receipt.replyId=reply.id;
  receipt.checks=['known isolated database','real persisted QA human post','immediate agent reply in human community','visible AI label','current reply JSON and count','hosted thread HTML','no pending review','unauthorized removal denied','remove and restore reply','revoked agent denied'];
  receipt.status='passed';
} catch(error) {
  receipt.reason=error instanceof assert.AssertionError?error.message:'Hosted verification failed; credential and provider payloads omitted';process.exitCode=1;
} finally {client?.close();console.log(JSON.stringify(receipt));}
