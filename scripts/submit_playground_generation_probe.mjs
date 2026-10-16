/** One fixed operator generation; exclusive start receipt prevents blind retries. */
import { readFile,writeFile,mkdir,access } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if(!args.cli||!args.receipt||!args.out)throw Error('Supply --cli, --receipt and a fresh --out.');
const out=path.resolve(args.out),root=path.resolve('artifacts/playground-authenticated-20261015');
if(!out.startsWith(root+path.sep))throw Error('Use a versioned diagnostic artifact directory.');
const deployment=JSON.parse(await readFile(args.receipt,'utf8'));
if(deployment.target!=='preview'||!/^https:\/\/eidos-sentinel-[a-z0-9]+-1brentbm-1876s-projects\.vercel\.app$/.test(deployment.url))throw Error('Unexpected diagnostic deployment.');
await mkdir(out,{recursive:true});
try{await access(path.join(out,'generation-submission-started.json'));throw Error('A generation was already started. Read the ledger; do not retry.');}catch(error){if(error.code!=='ENOENT')throw error;}
const {env}=JSON.parse(await readFile('apps/sentinel-lab/.vercel/playground-manual.private.json','utf8'));
await writeFile(path.join(out,'generation-submission-started.json'),JSON.stringify({timestampUtc:new Date().toISOString(),deploymentId:deployment.id,mode:'one fixed public Sol sample',automaticRetry:false},null,2)+'\n',{flag:'wx'});
const start=Date.now(),result=spawnSync(process.execPath,[args.cli,'curl','/api/playground-provider-generation','--deployment',deployment.url,'--scope','team_TEsnMvKU6SdbYjVhws6Ttp8H','--','--request','POST','--header','@-','--data-binary','{}','--silent','--show-error','--write-out','\n%{http_code}'],{
  cwd:'apps/sentinel-lab',encoding:'utf8',timeout:150000,
  input:['x-eidos-platform-token: '+env.EIDOS_PLATFORM_TOKEN,'x-eidos-site-origin: https://eidosworks-playground-20261014.pages.dev','content-type: application/json'].join('\n')+'\n',
  env:{...process.env,VERCEL_PROJECT_ID:'prj_5aL0COF1DONfPKI4fv25NitAdkOE',VERCEL_ORG_ID:'team_TEsnMvKU6SdbYjVhws6Ttp8H'},
});
const lines=(result.stdout||'').trim().split(/\r?\n/),status=Number(lines.pop());
let value={};try{value=JSON.parse(lines.join('\n'));}catch{/* Unknown outcome; no retry. */}
const receipt={timestampUtc:new Date().toISOString(),deploymentId:deployment.id,sourceCommit:deployment.sourceCommit,bundleHash:deployment.bundleHash,
  status:Number.isInteger(status)?status:null,cliExit:result.status,clientRoundTripMs:Date.now()-start,automaticRetry:false,
  providerKeyReadLocally:false,secretValuesPrinted:false,provider:value};
receipt.passed=result.status===0&&status===200&&value.state==='generated'&&value.usable===true&&value.generationRequests===1&&value.countRequests===0&&
  value.sourceCommit===deployment.sourceCommit&&value.bundleHash===deployment.bundleHash;
await writeFile(path.join(out,'generation-http-receipt.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({passed:receipt.passed,status:receipt.status,state:value.state||null,providerRequestId:value.providerRequestId||null,providerCostMicros:value.providerCostMicros??null,providerRoundTripMs:value.providerRoundTripMs??null}));
if(!receipt.passed)process.exitCode=1;
