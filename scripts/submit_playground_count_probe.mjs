/** One explicitly approved fixed-input count. A local start receipt prevents blind resubmission. */
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if(!args.cli||!args.receipt||!args.out)throw Error('Supply --cli, --receipt and a fresh --out artifact directory.');
const out=path.resolve(args.out),root=path.resolve('artifacts/playground-authenticated-20261015');
if(!out.startsWith(root+path.sep))throw Error('Use a versioned public evidence directory.');
const deployment=JSON.parse(await readFile(args.receipt,'utf8'));
if(deployment.target!=='preview'||!/^https:\/\/eidos-sentinel-[a-z0-9]+-1brentbm-1876s-projects\.vercel\.app$/.test(deployment.url))throw Error('Unexpected diagnostic deployment.');
await mkdir(out,{recursive:true});
try{await access(path.join(out,'count-submission-started.json'));throw Error('A count submission was already started. Read the durable ledger; do not retry.');}catch(error){if(error.code!=='ENOENT')throw error;}
const {env}=JSON.parse(await readFile('apps/sentinel-lab/.vercel/playground-manual.private.json','utf8'));
await writeFile(path.join(out,'count-submission-started.json'),JSON.stringify({timestampUtc:new Date().toISOString(),deploymentId:deployment.id,mode:'one owner-approved fixed public Sol sample',automaticRetry:false},null,2)+'\n',{flag:'wx'});
const started=Date.now();
const result=spawnSync(process.execPath,[args.cli,'curl','/api/playground-provider-check','--deployment',deployment.url,'--scope','team_TEsnMvKU6SdbYjVhws6Ttp8H','--','--request','POST','--header','@-','--data-binary','{}','--silent','--show-error','--write-out','\n%{http_code}'],{
  cwd:'apps/sentinel-lab',encoding:'utf8',timeout:60000,
  input:['x-eidos-platform-token: '+env.EIDOS_PLATFORM_TOKEN,'x-eidos-site-origin: https://eidosworks-playground-20261014.pages.dev','content-type: application/json'].join('\n')+'\n',
  env:{...process.env,VERCEL_PROJECT_ID:'prj_5aL0COF1DONfPKI4fv25NitAdkOE',VERCEL_ORG_ID:'team_TEsnMvKU6SdbYjVhws6Ttp8H'},
});
const lines=(result.stdout||'').trim().split(/\r?\n/),status=Number(lines.pop());
let value={};try{value=JSON.parse(lines.join('\n'));}catch{/* Preserve uncertainty; suppress raw output. */}
const ascii=x=>typeof x==='string'&&/^[!-~]{1,200}$/.test(x)?x:null;
const receipt={timestampUtc:new Date().toISOString(),deploymentId:deployment.id,sourceCommit:deployment.sourceCommit,bundleHash:deployment.bundleHash,
  status:Number.isInteger(status)?status:null,cliExit:result.status,clientRoundTripMs:Date.now()-started,
  model:value.model==='gpt-6.1-sol'?value.model:null,state:ascii(value.state),providerStatus:Number.isInteger(value.providerStatus)?value.providerStatus:null,
  inputTokens:Number.isSafeInteger(value.inputTokens)&&value.inputTokens>=0?value.inputTokens:null,
  providerRequestId:ascii(value.providerRequestId),clientRequestId:ascii(value.clientRequestId),requestHash:/^[a-f0-9]{64}$/.test(value.requestHash||'')?value.requestHash:null,
  providerHeldMicros:Number.isSafeInteger(value.providerHeldMicros)?value.providerHeldMicros:null,billing:'unknown',
  generationRequests:value.generationRequests===0?0:null,automaticRetry:false,providerKeyReadLocally:false,secretValuesPrinted:false};
receipt.passed=result.status===0&&status===200&&receipt.state==='counted'&&receipt.model==='gpt-6.1-sol'&&receipt.inputTokens!==null&&receipt.providerHeldMicros===5000000&&receipt.generationRequests===0;
await writeFile(path.join(out,'count-http-receipt.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));if(!receipt.passed)process.exitCode=1;
