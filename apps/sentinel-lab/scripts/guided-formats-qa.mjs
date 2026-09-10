import { createRequire } from "node:module";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || "playwright");
const root=resolve(import.meta.dirname,"../../..");
const out=resolve(root,`artifacts/sentinel-guided-20260908/${process.env.EIDOS_QA_TAG || 'formats'}`); mkdirSync(out,{recursive:true});
const base=process.env.EIDOS_QA_URL || "http://127.0.0.1:3210";
const access=process.env.EIDOS_QA_ACCESS_FILE ? JSON.parse(readFileSync(process.env.EIDOS_QA_ACCESS_FILE,"utf8")).url : null;
const keys=JSON.parse(readFileSync(resolve(root,"artifacts/sentinel-guided-private/local-access.json"),"utf8"));
const browser=await chromium.launch({headless:true,...(process.env.CHROME_BIN?{executablePath:process.env.CHROME_BIN}:{})});
const context=await browser.newContext({viewport:{width:1365,height:950},reducedMotion:"reduce"});
const page=await context.newPage();
const receipt={base,startedAt:new Date().toISOString(),checks:[],errors:[]};
page.on('pageerror',e=>receipt.errors.push(e.message));
async function request(path,body,key=keys.alice,method=body===undefined?'GET':'POST') {
  const response=await context.request.fetch(`${base}/api/lab/v1/${path}`,{method,headers:{Authorization:`Bearer ${key}`,'Idempotency-Key':crypto.randomUUID().replaceAll('-','_')},...(body===undefined?{}:{data:body})});
  return {status:response.status(),body:await response.json()};
}
async function untilJob(id) {
  for(let i=0;i<160;i++) {const r=await request(`jobs/${id}`);if(r.status!==200)throw Error(`Job status HTTP ${r.status}`); if(['completed','failed','cancelled','expired'].includes(r.body.status))return r.body; if(r.body.status==='queued')await request(`jobs/${id}/resume`,{}); await new Promise(r=>setTimeout(r,2000));}
  throw Error('Job did not finish in bounded polling window');
}
async function datasetFlow(filename,{documents=false,url=false,temporal=true}={}) {
  await page.getByRole('button',{name:/^01Add data$/}).click();
  if(url){await page.getByLabel('Public HTTPS URL',{exact:true}).fill(filename);await page.getByRole('button',{name:'Import source',exact:false}).click();}
  else await page.getByLabel('Upload data file').setInputFiles(resolve(root,'artifacts/sentinel-guided-20260908/fixtures',filename));
  await page.getByRole('heading',{name:'Confirm understanding',exact:true}).waitFor({timeout:300000});
  if(!documents){
    await page.getByLabel('One row represents',{exact:true}).fill('One synthetic measurement record for adapter validation');
    if(temporal){await page.getByLabel('Timestamp',{exact:true}).selectOption('timestamp');await page.getByLabel('Entity key',{exact:true}).selectOption('host');}
    if(await page.getByLabel('latency_ms units',{exact:true}).count())await page.getByLabel('latency_ms units',{exact:true}).fill('ms');
    if(await page.getByLabel('load units',{exact:true}).count())await page.getByLabel('load units',{exact:true}).fill('%');
    // Log sequence numbers are identifiers, explicitly exclude from the model.
    if(filename.endsWith('.log') && await page.getByRole('checkbox',{name:'request',exact:true}).count())await page.getByRole('checkbox',{name:'request',exact:true}).uncheck();
  }
  await page.getByRole('button',{name:'Confirm interpretation',exact:false}).click();
  await page.getByRole('heading',{name:'Analyze',exact:true}).waitFor({timeout:300000});
  if(!documents && !temporal)await page.getByRole('radio',{name:'Explore patterns',exact:false}).check();
  await page.getByRole('button',{name:'Run analysis',exact:false}).click();
  await page.getByRole('heading',{name:'Investigate',exact:true}).waitFor({timeout:300000});
  const view=page.getByRole('button',{name:/^View [rp]/}).first();
  if(await view.count()){await view.click();await page.getByText(/Source SHA-256/).waitFor();}
  const downloading=page.waitForEvent('download');await page.getByRole('button',{name:'Download results',exact:false}).click();
  const file=await downloading;const safe=filename.split('/').pop().replaceAll(/[^\w.-]/g,'_');await file.saveAs(resolve(out,`${safe}-result.json`));
  const result=JSON.parse(readFileSync(resolve(out,`${safe}-result.json`),'utf8'));
  if(!result.receipt?.processIsolation || !result.inputSha256)throw Error('Missing execution receipt');
  if(!documents && !temporal && !result.patterns?.pairs.length)throw Error('Pattern association evidence missing');
  if(documents){
    if(!result.encoder?.includes('c9745ed')||result.vectors[0].length!==384)throw Error('Real semantic encoder missing');
    await page.getByLabel('Question',{exact:true}).fill('What evidence describes database delays after the cache was unavailable?');
    await page.getByRole('button',{name:'Find supporting evidence',exact:true}).click();
    await page.getByText('semantic retrieval; deterministic extractive answer',{exact:true}).waitFor({timeout:300000});
    await page.getByRole('button',{name:'Inspect source',exact:true}).first().click();
    for(const path of [`runs/${result.id}`,`datasets/${result.datasetId}?record=${result.passageIds[0]}`])if((await request(path,undefined,keys.bob)).status!==404)throw Error('Private semantic data crossed owners');
  }
  await page.screenshot({path:resolve(out,`${safe}.png`),fullPage:true});
  receipt.checks.push({name:filename,status:'passed',runId:result.id,datasetId:result.datasetId,method:result.method,inputSha256:result.inputSha256,records:result.metrics.rows || result.metrics.evaluatedForecasts || result.metrics.passages});
  console.log(`${filename} browser -> parser -> analysis -> persisted result -> source passed`);
}
try {
  if(access)await page.goto(access,{waitUntil:'domcontentloaded',timeout:60000});
  await page.goto(base,{waitUntil:'networkidle'});await page.getByLabel('Access key',{exact:true}).fill(keys.alice);await page.getByRole('button',{name:'Sign in',exact:true}).click();await page.getByText('Signed in · pilot access').waitFor();
  const names=(process.env.EIDOS_QA_FILES || 'service-latency.xlsx,service-latency.parquet,service-latency.json,service-latency.jsonl,service.log,incident-notes.txt,incident-notes.html,incident-notes.pdf,plain.log').split(',');
  for(const name of names.filter(n=>n!=='URL_ONLY'))await datasetFlow(name,{documents:name.startsWith('incident-')||name==='plain.log'});
  await page.getByRole('button',{name:/^01Add data$/}).click();
  await page.getByLabel('Upload data file').setInputFiles(resolve(root,'artifacts/sentinel-guided-20260908/fixtures/oversize.csv'));await page.getByRole('alert').filter({hasText:'2 MB'}).waitFor();
  receipt.checks.push({name:'oversize browser rejection',status:'passed'});await page.getByRole('button',{name:'Dismiss',exact:true}).click();
  await page.getByLabel('Upload data file').setInputFiles(resolve(root,'artifacts/sentinel-guided-20260908/fixtures/malformed.json'));await page.getByRole('alert').filter({hasText:'JSONDecodeError'}).waitFor({timeout:180000});
  receipt.checks.push({name:'malformed parser failure visible in browser',status:'passed'});await page.getByRole('button',{name:'Dismiss',exact:true}).click();
  for(const url of ['https://127.0.0.1/internal','https://169.254.169.254/latest/meta-data/','https://httpbin.org/redirect-to?url=https%3A%2F%2F127.0.0.1%2Finternal']){
    const r=await request('imports',{url});if(r.status<400)throw Error('Unsafe URL accepted');receipt.checks.push({name:`URL rejection ${new URL(url).hostname}`,status:'passed',http:r.status,message:r.body.error});
  }
  await datasetFlow('https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv',{url:true,temporal:false});
  await datasetFlow('https://example.com/',{url:true,documents:true});
  if(receipt.errors.length)throw Error('Browser runtime errors');receipt.status='passed';
}catch(error){receipt.status='failed';receipt.failure=error.message;process.exitCode=1;await page.screenshot({path:resolve(out,'failure.png'),fullPage:true}).catch(()=>{});}
finally{receipt.finishedAt=new Date().toISOString();writeFileSync(resolve(out,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));await browser.close();}
