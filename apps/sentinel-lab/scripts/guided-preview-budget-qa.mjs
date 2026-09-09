import {createRequire} from 'node:module';import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';import {resolve} from 'node:path';
const require=createRequire(import.meta.url),{chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=resolve(import.meta.dirname,'../../..'),base=process.env.EIDOS_QA_URL;
const keys=JSON.parse(readFileSync(resolve(root,'artifacts/sentinel-guided-private/local-access.json')));
const out=resolve(root,'artifacts/sentinel-guided-20260908/preview-budget');mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.CHROME_BIN?{executablePath:process.env.CHROME_BIN}:{})}),context=await browser.newContext();
let monitor,ingestKey;const receipt={base,startedAt:new Date().toISOString(),checks:[],hostedProcessing:'blocked: HTTP 402 payment_required; no simulated outcome'};
const check=(name,ok)=>{if(!ok)throw Error(name);receipt.checks.push({name,status:'passed'});};
async function api(path,data,key=keys.bob,intent=crypto.randomUUID().replaceAll('-','_')){const r=await context.request.fetch(`${base}/api/lab/v1/${path}`,{method:data===undefined?'GET':'POST',headers:{Authorization:`Bearer ${key}`,'Idempotency-Key':intent},...(data===undefined?{}:{data})});return {status:r.status(),body:await r.json()};}
try{
 const page=await context.newPage();await page.goto(JSON.parse(readFileSync(process.env.EIDOS_QA_ACCESS_FILE)).url,{waitUntil:'domcontentloaded'});
 const created=await api('monitors',{name:'Synthetic preview ingress acceptance',target:'service.latency',unit:'ms',synthetic:true,autoProcess:false});check('hosted private monitor created',created.status===201);({monitor,key:ingestKey}=created.body);
 writeFileSync(resolve(root,'artifacts/sentinel-guided-private/preview-source-key.json'),JSON.stringify({key:ingestKey}));
 const events=[0,1,2].map(i=>({id:`preview-ingress-${i}`,eventTime:`2026-08-01T00:0${i}:00Z`,value:40+i,attributes:{synthetic:true}}));
 const first=await api(`telemetry/${monitor.id}`,{events},ingestKey);check('real HTTPS ingress stores original points',first.status===200&&first.body.accepted===3&&first.body.lastOffset===3);
 check('identical retries preserve offsets',(await api(`telemetry/${monitor.id}`,{events},ingestKey)).body.duplicates===3);
 check('changed duplicate rejects whole batch',(await api(`telemetry/${monitor.id}`,{events:[{...events[0],value:99}]},ingestKey)).status===409);
 check('ingest key cannot read private results',(await api(`runs`,undefined,ingestKey)).status===403);
 check('other owner cannot read source',(await api(`monitors/${monitor.id}`,undefined,keys.alice)).status===404);
 const otlp={resourceMetrics:[{resource:{attributes:[{key:'service.name',value:{stringValue:'synthetic-preview'}}]},scopeMetrics:[{metrics:[{name:'service.latency',gauge:{dataPoints:[{timeUnixNano:'1785542580000000000',asDouble:43}]}}]}]}]};
 check('hosted OTLP JSON endpoint accepts a scoped metric',(await api(`telemetry/${monitor.id}/v1/metrics`,otlp,ingestKey)).status===200);
 const state=(await api(`monitors/${monitor.id}`)).body;check('ingress remains distinct from processed model evidence',state.processedOffset===0&&state.buffered===4&&state.warmupRemaining===24);
 writeFileSync(resolve(out,'monitor-state.json'),JSON.stringify(state,null,2));
 await page.goto(base);await page.getByLabel('Access key',{exact:true}).fill(keys.bob);await page.getByRole('button',{name:'Sign in',exact:true}).click();await page.getByText('Signed in · pilot access').waitFor();await page.getByRole('button',{name:'Monitors',exact:true}).click();await page.getByRole('heading',{name:monitor.name,exact:true}).waitFor();await page.screenshot({path:resolve(out,'monitor-desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});await page.screenshot({path:resolve(out,'monitor-mobile.png'),fullPage:true});check('hosted monitor fits mobile viewport',!await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1));
 // One genuine allocation request verifies the observed external rejection.
 const rejected=await api(`monitors/${monitor.id}/process`,{});receipt.allocation=rejected.body;
 check('billing rejection is explicit and terminal',rejected.body.status==='failed'&&rejected.body.error.includes('HTTP 402')&&rejected.body.error.includes('payment_required'));
 check('rejected job cannot appear as a successful run',(await api(`runs/${rejected.body.id}`)).status===404);
 receipt.monitorId=monitor.id;receipt.status='passed';
}catch(e){receipt.status='failed';receipt.failure=e.message;process.exitCode=1;}
finally{if(monitor){const revoke=await api(`monitors/${monitor.id}/revoke`,{});if(revoke.status===200)check('test source credential revoked',(await api(`telemetry/${monitor.id}`,{events:[{id:'revoked',eventTime:'2026-08-01T00:05:00Z',value:1}]},ingestKey)).status===401);}receipt.finishedAt=new Date().toISOString();writeFileSync(resolve(out,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));await browser.close();}
