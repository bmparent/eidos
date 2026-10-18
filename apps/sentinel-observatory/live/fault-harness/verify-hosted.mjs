/** Owner-private acceptance. Credentials enter hidden stdin, never files/argv. */
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,join} from 'node:path';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const arg=name=>process.argv.includes(name)?process.argv[process.argv.indexOf(name)+1]:undefined;
const origin='https://sentinel-evidence-observatory.bmparent.chatgpt.site',out=resolve(arg('--out') ?? 'artifacts/proof_runs/2026-10-18/observatory-m1/hosted');mkdirSync(out,{recursive:true});
process.stdin.setRawMode?.(true);process.stdin.resume();console.log('Ready for owner-private verification JSON on stdin (input is hidden).');
const credentials=await new Promise((accept,reject)=>{let text='';const data=chunk=>{text+=chunk.toString();if(text.includes('\n')){process.stdin.off('data',data);process.stdin.setRawMode?.(false);process.stdin.pause();try{accept(JSON.parse(text.trim()));}catch{reject(new Error('Invalid hidden input'));}}};process.stdin.on('data',data);});
const hash=bytes=>createHash('sha256').update(bytes).digest('hex'),headers={'OAI-Sites-Authorization':`Bearer ${credentials.site_token}`,Origin:origin},receipts=[];
async function call(path,body,extra={}){const response=await fetch(origin+path,{method:body===undefined?'GET':'POST',headers:{...headers,...(body?{'Content-Type':'application/json'}:{}),...extra},body:body===undefined?undefined:JSON.stringify(body),redirect:'manual',signal:AbortSignal.timeout(20000)});return {response,bytes:Buffer.from(await response.arrayBuffer())};}
const capability=await call('/api/capabilities');assert.equal(capability.response.status,200);const caps=JSON.parse(capability.bytes);
writeFileSync(join(out,'capabilities.json'),JSON.stringify(caps,null,2)+'\n');
for(const [url,file] of [['/probe-explanation.mjs','probe-explanation.mjs'],['/target-panel.mjs','web/target-panel.mjs'],['/app.js','web/app.js'],['/styles.css','web/styles.css']]){
  const result=await call(url),local=Buffer.from(readFileSync(new URL('../'+file,import.meta.url),'utf8').replaceAll('\r\n','\n'));
  assert.equal(result.response.status,200);const match=hash(result.bytes)===hash(local);receipts.push({check:'served_source',route:url,sha256:hash(result.bytes),expected_sha256:hash(local),passed:match});
}
const anonymous=await fetch(origin+'/api/capabilities',{redirect:'manual',signal:AbortSignal.timeout(10000)});receipts.push({check:'anonymous_denied',status:anonymous.status,passed:anonymous.status!==200});
for(const [name,body] of [['loopback',{target:'127.0.0.1'}],['metadata',{target:'169.254.169.254'}],['private_ipv6',{target:'http://[::1]/'}],['credentials',{target:'https://user:pass@example.com/'}],['port',{target:'https://example.com:8080/'}],['recursive',{target:origin}],['forged_tenant',{target:'https://example.com/',tenant_id:'forged'}]]){
  const result=await call('/api/observe',body);receipts.push({check:name,status:result.response.status,passed:result.response.status>=400&&result.response.status<500});
}
const missing=await call('/api/interpret',{evidence_id:'evt_'+'f'.repeat(32)});receipts.push({check:'missing_evidence_no_model_dispatch',status:missing.response.status,passed:missing.response.status===404});
const cross=await call('/api/interpret',{evidence_id:'evt_'+'f'.repeat(32)},{Origin:'https://other.example'});receipts.push({check:'cross_origin_denied',status:cross.response.status,passed:cross.response.status===403});
if(arg('--public-test')==='example.com'){
  const result=await call('/api/observe',{target:'https://example.com/'}),payload=JSON.parse(result.bytes);
  writeFileSync(join(out,'public-test-observation.json'),JSON.stringify(payload,null,2)+'\n');
  const event=payload.event;receipts.push({check:'one_explicit_public_test',target:'https://example.com/',status:result.response.status,contract:event?.measurement_contract ?? null,evidence_id:event?.id ?? null,storage:event?.target_storage ?? null,passed:result.response.status===200&&event?.measurement_contract==='target-evidence-v2'&&event?.target_storage==='retained'});
  if(event){const history=await call('/api/target-history?target_id='+event.source_id),rows=JSON.parse(history.bytes);receipts.push({check:'hosted_retention',passed:rows.events?.some(x=>x.id===event.id)===true});}
}
const report={schema:'private-hosted-acceptance-v1',timestamp:new Date().toISOString(),origin,version:caps.version,receipts,passed:receipts.every(x=>x.passed),actual_model_calls:0,two_tenant_test:'not tested; owner-private service access is not tenant identity',continuous_pilot:'not started'};
if(arg('--playwright-module')){
  const {chromium}=createRequire(import.meta.url)(arg('--playwright-module')),browser=await chromium.launch({headless:true,executablePath:arg('--browser')});
  report.browser=[];
  try{for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
    const context=await browser.newContext({viewport,reducedMotion:'reduce'}),page=await context.newPage(),errors=[];
    await context.route('**/*',route=>new URL(route.request().url()).origin===origin?route.continue({headers:{...route.request().headers(),...headers}}):route.abort());
    page.on('pageerror',error=>errors.push(error.message));page.on('console',msg=>{if(msg.type()==='error')errors.push(msg.text());});
    await page.goto(origin,{waitUntil:'networkidle'});assert.match(await page.title(),/Sentinel/);
    await page.locator('#connect-target').focus();assert.equal(await page.evaluate(()=>document.activeElement?.id),'connect-target');
    await page.locator('#coverage-open').click();assert.equal(await page.locator('#coverage-dialog').evaluate(el=>el.open),true);await page.keyboard.press('Escape');
    assert.equal(await page.locator('#coverage-dialog').evaluate(el=>el.open),false);
    await page.screenshot({path:join(out,`private-${viewport.width}.png`),fullPage:true});
    const layout=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));assert.ok(layout.scroll<=layout.width+1);assert.deepEqual(errors,[]);
    report.browser.push({viewport,url:origin,empty_authenticated_page:true,keyboard_and_coverage_dialog:'passed',console_errors:errors,layout,physical_device:'not tested'});await context.close();
  }}finally{await browser.close();}
}
writeFileSync(join(out,'receipt.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({version:caps.version,passed:report.passed,checks:receipts.length,failed:receipts.filter(x=>!x.passed).map(x=>x.check)}));if(!report.passed)process.exitCode=1;
