import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
const require=createRequire(import.meta.url),{chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=resolve(import.meta.dirname,'../../..'),base=process.env.EIDOS_QA_URL || 'http://127.0.0.1:3210';
const keys=JSON.parse(readFileSync(resolve(root,'artifacts/sentinel-guided-private/local-access.json')));
const out=resolve(root,'artifacts/sentinel-guided-20260908/reviewed-history');mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.CHROME_BIN?{executablePath:process.env.CHROME_BIN}:{})}),context=await browser.newContext();
const receipt={base,startedAt:new Date().toISOString(),checks:[]};
function check(name,ok){if(!ok)throw Error(name);receipt.checks.push({name,status:'passed'});}
async function api(path,data,key=keys.alice){const r=await context.request.fetch(`${base}/api/lab/v1/${path}`,{method:data===undefined?'GET':'POST',headers:{Authorization:`Bearer ${key}`,'Idempotency-Key':crypto.randomUUID().replaceAll('-','_')},...(data===undefined?{}:{data})});return {status:r.status(),body:await r.json()};}
try{
 if(process.env.EIDOS_QA_ACCESS_FILE){const p=await context.newPage();await p.goto(JSON.parse(readFileSync(process.env.EIDOS_QA_ACCESS_FILE)).url,{waitUntil:'domcontentloaded'});await p.close();}
 const all=(await api('runs')).body.filter(r=>r.method.includes('Torch'));if(all.length<2)throw Error('Run CSV browser acceptance twice before historical-review acceptance');
 const current=(await api(`runs/${all[0].id}`)).body,other=(await api(`runs/${all[1].id}`)).body;
 const f=current.findings[0],match=other.findings.find(g=>g.entity===f.entity&&g.detector===f.detector);if(!match)throw Error('Two related synthetic findings required');
 const review=await api(`runs/${other.id}/feedback`,{findingId:match.id,review:'useful',note:'Synthetic acceptance review; not a production incident'});check('historical review saved with provenance',review.status===201&&!!review.body.reviewedAt);
 const answer=(await api(`runs/${current.id}/questions`,{question:'Has this happened before?',findingId:f.id})).body;
 const historical=answer.similarHistoricalEvents.find(h=>h.runId===other.id&&h.findingId===match.id);
 check('related history includes its actual reviewed outcome',historical?.reviewedOutcomes.some(r=>r.note==='Synthetic acceptance review; not a production incident'&&r.review==='useful'));
 check('review does not mutate model findings or frozen metrics',JSON.stringify((await api(`runs/${other.id}`)).body)===JSON.stringify(other));
 check('second user cannot query private reviewed history',(await api(`runs/${current.id}/questions`,{question:'Has this happened before?'},keys.bob)).status===404);
 writeFileSync(resolve(out,'answer.json'),JSON.stringify(answer,null,2));receipt.status='passed';
}catch(e){receipt.status='failed';receipt.failure=e.message;process.exitCode=1;}
finally{receipt.finishedAt=new Date().toISOString();writeFileSync(resolve(out,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));await browser.close();}
