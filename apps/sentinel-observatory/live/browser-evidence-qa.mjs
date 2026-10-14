/** Rendered fixture acceptance. Run from repo root; no provider calls. */
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const modulePath=process.env.EIDOS_PLAYWRIGHT_MODULE;
const {chromium}=await import(modulePath?pathToFileURL(modulePath).href:'playwright');
const origin=process.env.EIDOS_OBSERVATORY_QA_URL||'http://127.0.0.1:8770';
const out=path.resolve('artifacts/showcase-observatory-20261013');
await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,channel:process.env.EIDOS_BROWSER_CHANNEL||undefined,args:['--disable-gpu']});
const receipt={environment:{origin,browser:await browser.version(),engine:'Chromium',gpu:'disabled deliberately; SVG/HTML interface',sourceMode:'explicit synthetic fixture',physicalDevice:false},checks:[],consoleErrors:[],screenshots:[]};
const record=(name,details={})=>receipt.checks.push({name,status:'passed',...details});
const fixture=[
 {timestamp:'2026-10-13T12:00:00Z',event_type:'flow',flow_id:11,src_ip:'192.0.2.10',dest_ip:'198.51.100.20',src_port:44321,dest_port:443,proto:'TCP',flow:{bytes_toserver:210,bytes_toclient:820,pkts_toserver:3,pkts_toclient:4,age:1}},
 {timestamp:'2026-10-13T12:01:00Z',event_type:'alert',flow_id:12,src_ip:'192.0.2.10',dest_ip:'198.51.100.20',proto:'TCP',alert:{signature_id:900001,signature:'QA fixture rule · not a real incident',severity:3,action:'allowed'}},
 {timestamp:'2026-10-13T12:02:00Z',event_type:'dns',flow_id:13,src_ip:'192.0.2.11',dest_ip:'198.51.100.53',proto:'UDP',dns:{rrname:'fixture.example',rrtype:'A',rcode:'NOERROR'}}
];
const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true,reducedMotion:'reduce'});
const page=await context.newPage();
page.on('pageerror',error=>receipt.consoleErrors.push(error.message));
page.on('console',message=>{if(message.type()==='error')receipt.consoleErrors.push(message.text());});
await page.route('**/evidence-qa',route=>route.fulfill({contentType:'text/html; charset=utf-8',body:'<!doctype html><html lang="en"><head><title>Sentinel coordinated evidence · fixture acceptance</title><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/styles.css"></head><body><main><h1>Observatory evidence preview</h1><p role="note">LABELED FIXTURE · synthetic acceptance data; no actual sensor traffic or model inference.</p><div id="qa"></div></main><script type="module">import {initializeNetworkPanel} from "/network-panel.mjs"; window.panel=initializeNetworkPanel(document.querySelector("#qa"));</script></body></html>'}));
try{
 await page.goto(`${origin}/evidence-qa`);await page.waitForFunction(()=>window.panel);
 assert.match(await page.title(),/Sentinel coordinated evidence/);record('page identity and meaningful rendered content');
 await page.evaluate(async rows=>{const {parseSecurityLog}=await import('/security-events.mjs');window.fixtureBatch=parseSecurityLog(rows.map(row=>JSON.stringify(row)).join('\n'),{sourceId:'qa-fixture',sourceName:'QA fixture · synthetic',format:'suricata'});window.panel.addSourceBatch(window.fixtureBatch,{id:'qa-fixture',name:'QA fixture · synthetic',mode:'fixture'});window.panel.addSourceBatch(window.fixtureBatch,{id:'qa-fixture',mode:'fixture'});},fixture);
 assert.equal(await page.locator('[data-part="records"]').textContent(),'3');record('duplicate batches retain exactly three records');
 assert.match(await page.locator('[data-part="badge"]').textContent(),/FIXTURE/);record('synthetic source visibly distinct from observed telemetry');
 await page.locator('.sn-event').filter({hasText:'QA fixture rule'}).click();
 const selection=await page.evaluate(()=>window.panel.getInterpretationContext().selection);
 assert.match(await page.locator('[data-part="inspector"]').textContent(),/900001/);
 assert.equal(await page.locator('.sn-series-mark.selected').count(),1);record('event selection coordinates list, series, topology and evidence drawer');
 await page.getByRole('button',{name:'Time series',exact:true}).click();
 assert.equal(await page.locator('[data-part="graph"]').isVisible(),false);
 await page.getByRole('button',{name:'Topology',exact:true}).click();
 assert.equal(await page.locator('[data-part="series"]').isVisible(),false);
 assert.equal((await page.evaluate(()=>window.panel.getInterpretationContext().selection)).event_id,selection.event_id);record('view switching preserves event selection');
 await page.getByRole('button',{name:'Overview',exact:true}).click();
 await page.evaluate(()=>window.panel.setTimeWindow(Date.parse('2026-10-13T12:00:30Z'),Date.parse('2026-10-13T12:01:30Z')));
 assert.equal(await page.locator('[data-part="records"]').textContent(),'1');assert.equal(await page.locator('.sn-event').count(),1);assert.equal(await page.locator('.sn-series-mark').count(),1);
 await page.evaluate(()=>window.panel.addSourceBatch(window.fixtureBatch,{id:'qa-fixture',mode:'fixture'}));
 assert.equal((await page.evaluate(()=>window.panel.getInterpretationContext().selection)).time_window.start,Date.parse('2026-10-13T12:00:30Z'));record('fixed window filters all views and survives incoming repeated receipts');
 assert.equal(await page.evaluate(()=>window.panel.setTimeWindow(100,0)),false);assert.equal(await page.locator('[data-part="records"]').textContent(),'1');record('invalid time range preserves the prior window');
 await page.getByRole('button',{name:'All retained',exact:true}).click();
 await page.locator('[data-node="192.0.2.10"]').focus();await page.keyboard.press('Enter');
 assert.equal(await page.locator('[data-part="records"]').textContent(),'2');assert.equal(await page.locator('.sn-event').count(),2);record('keyboard host selection synchronizes all views');
 await page.getByRole('button',{name:'Clear host filter'}).count().then(async count=>{if(count)await page.getByRole('button',{name:'Clear host filter'}).click();else await page.locator('[data-action="clear-node"]').click();});
 const downloadPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Export evidence',exact:true}).click();const download=await downloadPromise;const exportPath=path.join(out,'fixture-export.json');await download.saveAs(exportPath);const exported=JSON.parse(await fs.readFile(exportPath,'utf8'));assert.equal(exported.events.length,3);assert.equal(exported.sources[0].mode,'fixture');assert.equal(exported.selection.event_id,selection.event_id);record('downloaded evidence preserves source mode, receipts and selection',{file:'fixture-export.json'});
 const shot=path.join(out,'observatory-desktop-fixture.png');await page.screenshot({path:shot,fullPage:true});receipt.screenshots.push(shot);
 await page.setViewportSize({width:390,height:844});await page.getByRole('button',{name:'Time series',exact:true}).click();
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),true);record('390px reduced-motion fallback remains readable without page overflow');
 const phone=path.join(out,'observatory-phone-fixture.png');await page.screenshot({path:phone,fullPage:true});receipt.screenshots.push(phone);
 // Import through the real file input; malformed input must preserve retained evidence.
 await page.locator('[data-part="file"]').setInputFiles({name:'malformed.jsonl',mimeType:'application/json',buffer:Buffer.from('{broken')});await page.waitForFunction(()=>document.querySelector('[data-part="message"]').textContent.includes('No supported'));
 assert.equal((await page.evaluate(()=>window.panel.getEvidence())).events.length,3);record('malformed import preserves retained evidence');
 await page.locator('[data-part="file"]').setInputFiles({name:'fixture-import.jsonl',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(fixture[0]))});await page.waitForFunction(()=>window.panel.getEvidence().sources.length===2);assert.match(await page.locator('[data-part="sources"]').textContent(),/HISTORICAL IMPORT/);record('file import is explicitly historical beside synthetic fixture');
 await page.evaluate(id=>{window.panel.selectEvidence(id);window.panel.attachInterpretation(id,{summary:'Fixture model outage; original evidence remains available.',provenance:{model_ran:false,output_accepted:false,reason:'Fixture provider unavailable'}});},selection.event_id);
 assert.match(await page.locator('[data-part="inspector"]').textContent(),/MEASUREMENT SUMMARY/);assert.equal((await page.evaluate(()=>window.panel.getEvidence())).events.length,4);record('model outage retains original evidence and labels measured fallback');
 let feedCalls=0;
 await page.route('**/api/security-events',route=>{feedCalls++;return feedCalls===1?route.fulfill({status:403,contentType:'application/json',body:'{"error":"Fixture access denied"}'}):route.fulfill({contentType:'application/json',body:JSON.stringify({connected:true,source_id:'qa-feed-fixture',source_name:'Fixture feed · synthetic acceptance only',format:'suricata',mode:'live_log_tail',observed_at:new Date().toISOString(),text:JSON.stringify(fixture[0]),coverage:{scope:'Synthetic reconnect acceptance; not actual telemetry'}})});});
 await page.evaluate(()=>window.panel.setCapabilities({security_sensor:{connected:true,mode:'live_log_tail'},hosted_interpretation:{configured:false,connected:false,reason:'Fixture provider unavailable'}}));
 await page.locator('[data-action="stream"]').click();await page.waitForFunction(()=>document.querySelector('[data-part="badge"]').textContent.includes('UNAUTHORIZED'));
 assert.equal((await page.evaluate(()=>window.panel.getEvidence())).events.length,4);record('denied feed visibly unauthorized while retained evidence remains');
 await page.locator('[data-action="stream"]').click();await page.locator('[data-action="stream"]').click();await page.waitForFunction(()=>window.panel.getEvidence().sources.length===3);
 const before=(await page.evaluate(()=>window.panel.getEvidence())).events.length;
 await page.locator('[data-action="stream"]').click();await page.locator('[data-action="stream"]').click();await page.waitForFunction(()=>document.querySelector('[data-part="live"]').textContent.includes('Receiving'));
 assert.equal((await page.evaluate(()=>window.panel.getEvidence())).events.length,before);await page.locator('[data-action="stream"]').click();
 assert.match(await page.locator('[data-part="sources"]').textContent(),/PAUSED/);record('feed reconnect deduplicates records and pause preserves explicit coverage');
 // Actual app route remains intact; no provider or remote target request is issued.
 await page.goto(origin);await page.waitForSelector('#network-panel .sn-panel');assert.match(await page.title(),/Sentinel/i);record('unchanged app entry loads with coordinated network panel');
 const expected=receipt.consoleErrors.filter(message=>message==='Failed to load resource: the server responded with a status of 403 (Forbidden)');receipt.expectedConsoleErrors=expected;
 assert.deepEqual(receipt.consoleErrors.filter(message=>!expected.includes(message)),[]);record('no unexpected browser runtime or console errors',{expectedDeniedFeedResponses:expected.length});
 receipt.status='passed';
}catch(error){receipt.status='failed';receipt.error=error.stack;throw error;}finally{await fs.writeFile(path.join(out,'browser-acceptance.json'),JSON.stringify(receipt,null,2));await browser.close();}
console.log(JSON.stringify({status:receipt.status,checks:receipt.checks.length,receipt:path.join(out,'browser-acceptance.json')},null,2));
