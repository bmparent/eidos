import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
const require=createRequire(import.meta.url),{chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=resolve(import.meta.dirname,'../../..'),base=process.env.EIDOS_QA_URL || 'http://127.0.0.1:3210';
const out=resolve(root,`artifacts/sentinel-guided-20260908/${process.env.EIDOS_QA_TAG || 'stream'}`);mkdirSync(out,{recursive:true});
const privateDir=resolve(root,'artifacts/sentinel-guided-private');
const access=process.env.EIDOS_QA_ACCESS_FILE ? JSON.parse(readFileSync(process.env.EIDOS_QA_ACCESS_FILE,'utf8')).url : null;
const keys=JSON.parse(readFileSync(resolve(privateDir,'local-access.json')));
const browser=await chromium.launch({headless:true,...(process.env.CHROME_BIN?{executablePath:process.env.CHROME_BIN}:{})});
let context=await browser.newContext(),page;
async function authorizePreview(){if(access){const p=await context.newPage();await p.goto(access,{waitUntil:'domcontentloaded',timeout:60000});await p.close();}}
await authorizePreview();
const receipt={base,startedAt:new Date().toISOString(),checks:[]};
const check=(name,value=true)=>{if(!value)throw Error(name);receipt.checks.push({name,status:'passed'});console.log(name)};
const nap=ms=>new Promise(r=>setTimeout(r,ms));
async function api(path,body,key=keys.bob,expected=200){
 const r=await context.request.fetch(`${base}/api/lab/v1/${path}`,{method:body===undefined?'GET':'POST',headers:{Authorization:`Bearer ${key}`,'Idempotency-Key':crypto.randomUUID().replaceAll('-','_')},...(body===undefined?{}:{data:body})});
 const data=await r.json();if(expected && r.status()!==expected)throw Error(`${path}: ${r.status()} ${data.error || ''}`);return expected?data:{status:r.status(),data};
}
async function settle(job){for(let i=0;i<180;i++){job=await api(`jobs/${job.id}`);if(['completed','failed','cancelled','expired'].includes(job.status))return job;if(job.status==='queued')job=await api(`jobs/${job.id}/resume`,{},keys.bob);await nap(2000);}throw Error('bounded job polling expired')}
try{
 const {monitor,key}=await api('monitors',{name:'Synthetic connector acceptance',target:'service.latency',unit:'ms',synthetic:true,autoProcess:false},keys.bob,201);
 receipt.sourceId=monitor.id;
 const events=Array.from({length:60},(_,i)=>({id:`integration-${i}`,eventTime:new Date(Date.UTC(2026,7,1)+i*60000+(i>=45?600000:0)).toISOString(),value:i===40?100:40+Math.sin(i),attributes:{fixture:'synthetic'}}));events[55].eventTime=events[30].eventTime;
 const first=await api(`telemetry/${monitor.id}`,{events:events.slice(0,30)},key);check('first batch has contiguous acknowledged offsets',first.accepted===30&&first.lastOffset===30);
 const retry=await api(`telemetry/${monitor.id}`,{events:events.slice(0,30)},key);check('replay acknowledges duplicates without counting twice',retry.duplicates===30&&retry.lastOffset===30);
 const conflict=await api(`telemetry/${monitor.id}`,{events:[{...events[0],value:99}]},key,0);check('changed duplicate is rejected transactionally',conflict.status===409);
 check('ingest credential cannot read monitor configuration',(await api(`monitors/${monitor.id}`,undefined,key,0)).status===403);
 check('other owner cannot read source',(await api(`monitors/${monitor.id}`,undefined,keys.alice,0)).status===404);
 let job=await api(`monitors/${monitor.id}/process`,{},keys.bob,202);receipt.firstJobId=job.id;
 // Close the requesting client. An entirely new context recovers persisted work.
 await context.close();context=await browser.newContext();await authorizePreview();job=await settle(job);check('worker and durable result survive client interruption',job.status==='completed');
 let state=await api(`monitors/${monitor.id}`);check('warmup and first checkpoint persisted',state.processedOffset===30&&state.warmupRemaining===0);
 await api(`telemetry/${monitor.id}`,{events:events.slice(30)},key);
 job=await settle(await api(`monitors/${monitor.id}/process`,{},keys.bob,202));check('second batch resumes existing model',job.status==='completed');
 state=await api(`monitors/${monitor.id}`);check('late event and missing-time gap explicitly counted',state.processedOffset===60&&state.checkpoint.late===1&&state.checkpoint.gaps===1);
 check('injected event remains an anomaly',state.lastOutcome.outcomes.some(e=>e.recordId==='integration-40'&&e.anomaly));
 writeFileSync(resolve(out,'monitor-state.json'),JSON.stringify(state,null,2));
 // Reproduce all accepted stored events in a new Python process for exact state comparison.
 const full={events:[...state.events].sort((a,b)=>a.offset-b.offset),target:monitor.target,unit:monitor.unit,resetId:'initial',staleSeconds:300};
 writeFileSync(resolve(out,'full-replay-request.json'),JSON.stringify({...full,operation:'telemetry'}));
 if(!process.env.EIDOS_QA_URL){
  const python=resolve(root,process.platform==='win32'?'.venv-guided/Scripts/python.exe':'.venv-guided/bin/python');
  const r=spawnSync(python,['-m','sentinel_runner.guided.cli','--request',resolve(out,'full-replay-request.json'),'--out',resolve(out,'full-replay')],{cwd:root,env:{...process.env,PYTHONIOENCODING:'utf-8',EIDOS_CALLBACK_URL:''},encoding:'utf8',windowsHide:true});
  writeFileSync(resolve(out,'full-replay.log'),r.stdout+r.stderr);if(r.status!==0)throw Error('Direct replay failed');
  const replay=JSON.parse(readFileSync(resolve(out,'full-replay/result.json')));check('checkpoint-resume matches uninterrupted actual Torch replay',replay.stateSha256===state.lastOutcome.stateSha256);
 }
 const otlp={resourceMetrics:[{resource:{attributes:[{key:'service.name',value:{stringValue:'synthetic-acceptance'}}]},scopeMetrics:[{metrics:[{name:'service.latency',gauge:{dataPoints:[{timeUnixNano:String(BigInt(Date.UTC(2026,7,2))*1000000n),asDouble:42}]}}]}]}]};
 const otlpResponse=await api(`telemetry/${monitor.id}/v1/metrics`,otlp,key);check('real OTLP HTTP JSON endpoint accepts exact metric',Object.keys(otlpResponse).length===0);
 const reset=await api(`monitors/${monitor.id}/reset`,{reason:'Synthetic replay acceptance: explicit model reset'});check('reset archives prior state and restarts warmup',!!reset.resetId&&(await api(`monitors/${monitor.id}`)).warmupRemaining===24);
 // Fill exactly the remaining bounded buffer, then prove 429 leaves offsets intact.
 let current=await api(`monitors/${monitor.id}`),serial=0;
 while(current.buffered<1000){const n=Math.min(100,1000-current.buffered);await api(`telemetry/${monitor.id}`,{events:Array.from({length:n},()=>({id:`backpressure-${serial++}`,eventTime:'2026-08-03T00:00:00Z',value:1}))},key);current=await api(`monitors/${monitor.id}`);}
 const overflow=await api(`telemetry/${monitor.id}`,{events:[{id:'over-capacity',eventTime:'2026-08-03T00:00:01Z',value:1}]},key,0);check('bounded buffer returns backpressure with no partial insertion',overflow.status===429&&(await api(`monitors/${monitor.id}`)).buffered===1000);
 await api(`monitors/${monitor.id}/revoke`,{});check('revoked ingest key immediately rejected',(await api(`telemetry/${monitor.id}`,{events:[events[0]]},key,0)).status===401);
 page=await context.newPage();await page.goto(base);await page.getByLabel('Access key',{exact:true}).fill(keys.bob);await page.getByRole('button',{name:'Sign in',exact:true}).click();await page.getByText('Signed in · pilot access').waitFor();await page.getByRole('button',{name:'Monitors',exact:true}).click();await page.getByRole('heading',{name:monitor.name,exact:true}).waitFor();await page.screenshot({path:resolve(out,'monitor-desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});await page.screenshot({path:resolve(out,'monitor-mobile.png'),fullPage:true});check('monitor mobile layout fits viewport',!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1));
 receipt.status='passed';
}catch(e){receipt.status='failed';receipt.failure=e.message;process.exitCode=1;}finally{receipt.finishedAt=new Date().toISOString();writeFileSync(resolve(out,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));await browser.close();}
