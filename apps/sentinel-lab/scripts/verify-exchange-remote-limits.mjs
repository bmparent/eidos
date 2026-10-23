import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {randomUUID,randomBytes,createHash,createHmac} from 'node:crypto';
import {createClient} from '@libsql/client';
const s=JSON.parse(await readFile(process.env.EXCHANGE_QA_PRIVATE_FILE,'utf8'));
const base=new URL(process.env.EXCHANGE_QA_BASE_URL);
assert.match(base.hostname,/^eidos-exchange-qa-20261023-[a-z0-9]+-1brentbm-1876s-projects\.vercel\.app$/);
assert.equal(base.protocol,'https:');assert.equal(base.pathname,'/');assert.ok(!base.username&&!base.password&&!base.port&&!base.search&&!base.hash);
const dburl=new URL(s.EIDOS_EXCHANGE_TEST_TURSO_DATABASE_URL);
assert.equal(createHash('sha256').update(dburl.hostname+dburl.pathname).digest('hex'),'0599028d2df7969a4992e86be8971499744c42e83382a41df7cbe13b28f49c48');
const c=createClient({url:dburl.href,authToken:s.EIDOS_EXCHANGE_TEST_TURSO_AUTH_TOKEN});
const tag='qa-limit-'+randomUUID().slice(0,8), ids=[], threadIds=[];
const receipt={time:new Date().toISOString(),status:'failed',environment:'isolated-hosted-backend-explicit-database-fixtures',baseUrl:base.origin,checks:[],productionWrites:0,independentOperatorsObserved:0,quotaRestoration:'Exclusive protected QA database: restore exact pre-fixture quota rows; preserve all earlier real HTTP usage'};
let snapshot,stage='snapshot';
const q=async(sql,args=[])=>c.execute({sql,args});
const count=async t=>Number((await q('SELECT COUNT(*) n FROM '+t)).rows[0].n);
const day=Math.floor(Date.now()/86400000),hour=Math.floor(Date.now()/3600000);
const ip='192.0.2.193';
const visitor=createHmac('sha256',s.EIDOS_RATE_SECRET).update(new Date().toISOString().slice(0,10)+':'+ip).digest('hex');
const payload=name=>({name,description:'Explicit remote database limit fixture, not public acquisition.',rulesVersion:'2026-10-22',operatorAuthorized:true,acceptPublicContributions:true,discoverySource:'controlled'});
async function call(path,data,key,fixtureIp=ip){const r=await fetch(base.origin+'/api/works/v1'+path,{method:'POST',headers:{'content-type':'application/json','x-vercel-protection-bypass':s.EIDOS_PLATFORM_PREVIEW_BYPASS,'x-eidos-platform-token':s.EIDOS_PLATFORM_TOKEN,'x-eidos-site-origin':'https://qa.eidos-exchange-qa-20261023.pages.dev','x-eidos-client-ip':fixtureIp,...(key?{authorization:'Bearer '+key}:{})},body:JSON.stringify(data),redirect:'error',signal:AbortSignal.timeout(20000)});return {status:r.status,value:await r.json()};}
async function quota(bucket,used,period=day){await q('INSERT INTO eidos_quotas(bucket,period,used,expires) VALUES(?,?,?,?) ON CONFLICT(bucket,period) DO UPDATE SET used=excluded.used',[bucket,period,used,(period+2)*(period===hour?3600:86400)]);}
async function restore(){await c.batch([{sql:'DELETE FROM eidos_quotas',args:[]},...snapshot.map(r=>({sql:'INSERT INTO eidos_quotas(bucket,period,used,expires) VALUES(?,?,?,?)',args:[r.bucket,r.period,r.used,r.expires]}))],'write');}
async function denied(label,bucket,used,expected=429,period=day){stage=label;console.log('QA fixture: '+label);await restore();await quota(bucket,used,period);const before=await count('eidos_exchange_identities');assert.equal((await call('/api/community/agent-registration',payload(tag+'-denied'))).status,expected);assert.equal(await count('eidos_exchange_identities'),before);receipt.checks.push(label);}
try{
 snapshot=(await q('SELECT bucket,period,used,expires FROM eidos_quotas')).rows;
 await denied('global 200 daily registration-attempt ceiling','exchange-registration-attempts',200);
 await denied('per-IP 10 hourly registration-attempt ceiling','exchange-registration-attempt:'+visitor,10,429,hour);
 await denied('global 20 daily registration ceiling','exchange-registrations',20);
 await restore();stage='lifetime 100 includes revoked identities and concurrent boundary';console.log('QA fixture: '+stage);
 const before=await count('eidos_exchange_identities');assert.ok(before<99);
 const inserts=[];
 for(let i=before;i<99;i++){const id=randomUUID();ids.push(id);inserts.push({sql:'INSERT INTO eidos_exchange_identities(id,name,description,key_hash,created_at,expires_at,revoked) VALUES(?,?,?,?,?,?,1)',args:[id,tag+'-'+i,'Explicit revoked lifetime-cap fixture',createHash('sha256').update(randomBytes(32)).digest('hex'),new Date().toISOString(),0]});}
 await c.batch(inserts,'write');
 const race=await Promise.all(['192.0.2.194','192.0.2.195'].map((v,i)=>call('/api/community/agent-registration',payload(tag+'-race-'+i),undefined,v)));
 for(const r of race)if(r.status===201)ids.push(r.value.id);
 assert.deepEqual(race.map(r=>r.status).sort(),[201,409]);assert.equal(await count('eidos_exchange_identities'),100);receipt.checks.push(stage);
 if(ids.length)await c.batch(ids.map(id=>({sql:'DELETE FROM eidos_exchange_identities WHERE id=?',args:[id]})),'write');ids.length=0;
 await restore();stage='contribution fixture setup';
 const seed=async suffix=>{const id=randomUUID(),key='ew_exchange_'+randomBytes(32).toString('hex');ids.push(id);await q('INSERT INTO eidos_exchange_identities(id,name,description,key_hash,created_at,expires_at) VALUES(?,?,?,?,?,?)',[id,tag+'-'+suffix,'Explicit controlled quota test identity',createHash('sha256').update(key).digest('hex'),new Date().toISOString(),Math.floor(Date.now()/1000)+3600]);return{id,key};};
 const a=await seed('one'),b=await seed('other');
 const finding={title:'Controlled hosted quota boundary fixture',body:'Explicit test fixture. No independent visitor, customer or growth claim.'};
 for(const [label,bucket,limit] of [['identity 3 daily contributions','exchange-submission:'+a.id,3],['global 20 daily contributions','exchange-submissions',20]]){
  stage=label;console.log('QA fixture: '+label);await restore();await quota(bucket,limit);const n=await count('eidos_threads');assert.equal((await call('/api/community/agents',finding,a.key)).status,429);assert.equal(await count('eidos_threads'),n);receipt.checks.push(label);
 }
 await restore();stage='global pending queue 50 with concurrent writers';console.log('QA fixture: '+stage);
 const pending=Number((await q("SELECT (SELECT COUNT(*) FROM eidos_threads WHERE status='pending')+(SELECT COUNT(*) FROM eidos_replies WHERE status='pending') n")).rows[0].n);assert.ok(pending<49);
 const rows=[];for(let i=pending;i<49;i++){const id=randomUUID();threadIds.push(id);rows.push({sql:"INSERT INTO eidos_threads(id,title,body,category,author,author_type,owner_id,status,created_at) VALUES(?,?,?,'agents','Controlled QA fixture','agent',?,'pending',?)",args:[id,finding.title,finding.body,b.id,new Date().toISOString()]});}await c.batch(rows,'write');
 const overlap=await Promise.all([1,2].map(()=>call('/api/community/agents',finding,a.key)));for(const r of overlap)if(r.status===201)threadIds.push(r.value.id);
 assert.deepEqual(overlap.map(r=>r.status).sort(),[201,429]);receipt.checks.push(stage);
 receipt.status='passed';
}catch(e){receipt.failedCheck=stage;receipt.failure=e?.code||'assertion-or-provider-error';if(typeof e?.actual==='number')receipt.assertion={actual:e.actual,expected:e.expected};process.exitCode=1;
}finally{
 try{await q('DELETE FROM eidos_threads WHERE owner_id IN (SELECT id FROM eidos_exchange_identities WHERE name LIKE ?)',[tag+'-%']);if(threadIds.length)await c.batch(threadIds.map(id=>({sql:'DELETE FROM eidos_threads WHERE id=?',args:[id]})),'write');if(ids.length)await c.batch(ids.map(id=>({sql:'DELETE FROM eidos_exchange_identities WHERE id=?',args:[id]})),'write');await q('DELETE FROM eidos_exchange_identities WHERE name LIKE ?',[tag+'-%']);if(snapshot){await restore();assert.deepEqual((await q('SELECT bucket,period,used,expires FROM eidos_quotas ORDER BY bucket,period')).rows, snapshot.toSorted((a,b)=>a.bucket.localeCompare(b.bucket)||Number(a.period)-Number(b.period)));}receipt.fixturesRemoved=true;}catch{receipt.cleanupFailure=true;receipt.status='failed';process.exitCode=1;}
 c.close();await writeFile(process.env.EXCHANGE_QA_RECEIPT,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
}
