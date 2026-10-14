/** Private candidate QA. A scoped authorization receipt arrives through hidden stdin. */
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
if(process.stdin.isTTY)process.stdin.setRawMode(true);
process.stdout.write('Ready for private candidate QA JSON on stdin (input is hidden).\n');
const input=await new Promise((resolve,reject)=>{let line='';process.stdin.setEncoding('utf8');const read=chunk=>{line+=chunk;if(/[\r\n]/.test(line)){process.stdin.off('data',read);try{resolve(JSON.parse(line.trim()));}catch{reject(Error('Invalid QA input'));}}};process.stdin.on('data',read);});
if(process.stdin.isTTY)process.stdin.setRawMode(false);process.stdin.pause();
const {chromium}=await import(process.env.EIDOS_PLAYWRIGHT_MODULE?pathToFileURL(process.env.EIDOS_PLAYWRIGHT_MODULE).href:'playwright');
const browser=await chromium.launch({headless:true,channel:process.env.EIDOS_BROWSER_CHANNEL||undefined,args:['--disable-gpu']});
const out=path.resolve('artifacts/showcase-observatory-20261013'),receipt={url:input.url,source_mode:'hosted explicit fixture; no actual sensor/model claim',checks:[],console_errors:[],browser:await browser.version(),gpu:'disabled',screenshots:[]};
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',extraHTTPHeaders:{'OAI-Sites-Authorization':`Bearer ${input.bearer}`}});input.bearer=null;
const page=await context.newPage();page.on('pageerror',error=>receipt.console_errors.push(error.message));
try{
 const response=await page.goto(receipt.url+'/evidence-fixture');assert.equal(response.status(),200);await page.waitForSelector('.sn-event');assert.match(await page.title(),/labeled fixture/);receipt.checks.push('private hosted fixture renders');
 assert.equal(await page.locator('.sn-event').count(),3);await page.locator('.sn-event').filter({hasText:'Synthetic demonstration rule'}).click();assert.equal(await page.locator('.sn-series-mark.selected').count(),1);assert.match(await page.locator('[data-part="inspector"]').textContent(),/900001/);receipt.checks.push('hosted selection synchronizes list/series/drawer');
 await page.getByRole('button',{name:'Latest 15 minutes',exact:true}).click();const time=await page.locator('[data-part="window-status"]').textContent();await page.getByRole('button',{name:'Time series',exact:true}).click();assert.equal(await page.locator('[data-part="window-status"]').textContent(),time);receipt.checks.push('hosted time and receipt survive view change');
 const desktop=path.join(out,'hosted-observatory-desktop.png');await page.screenshot({path:desktop,fullPage:true});receipt.screenshots.push(desktop);
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);const phone=path.join(out,'hosted-observatory-phone.png');await page.screenshot({path:phone,fullPage:true});receipt.screenshots.push(phone);receipt.checks.push('390px reduced-motion no-GPU hosted fallback');
 const capabilities=await (await context.request.get(receipt.url+'/api/capabilities')).json();receipt.capabilities={sensor:capabilities.security_sensor,interpretation:capabilities.hosted_interpretation};assert.equal(capabilities.security_sensor.connected,false);assert.equal(capabilities.hosted_interpretation.configured,false);receipt.checks.push('isolated sensor/model bindings explicitly unavailable');
 const original=await context.request.get(receipt.url+'/api/security-events');receipt.sensor_status=original.status();receipt.sensor_response=await original.json();
 const anonymous=await browser.newContext();const denied=await anonymous.request.get(receipt.url+'/evidence-fixture');receipt.unauthenticated_status=denied.status();assert.ok([401,403].includes(denied.status()));await anonymous.close();receipt.checks.push('unauthenticated candidate access denied');
 assert.deepEqual(receipt.console_errors,[]);receipt.checks.push('no hosted runtime errors');receipt.status='passed';
}catch(error){receipt.status='failed';receipt.error=error.stack;throw error;}finally{await fs.writeFile(path.join(out,'hosted-browser-acceptance.json'),JSON.stringify(receipt,null,2));await browser.close();}
console.log(JSON.stringify({status:receipt.status,checks:receipt.checks.length,url:receipt.url}));
