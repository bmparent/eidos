/** Auth-only hosted acceptance. Private relay headers travel on stdin; keys are never read. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import assert from 'node:assert/strict';
const args = Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if (!args.cli || !args.receipt || !args.out) throw Error('Supply --cli, --receipt and a fresh --out artifact directory.');
const artifactRoot=path.resolve('artifacts/playground-authenticated-20261015'),out=path.resolve(args.out);
if (out!==artifactRoot&&!out.startsWith(artifactRoot+path.sep)) throw Error('Unsafe artifact destination.');
const deployment=JSON.parse(await readFile(args.receipt,'utf8'));
if (!/^https:\/\/eidos-sentinel-[a-z0-9]+-1brentbm-1876s-projects\.vercel\.app$/.test(deployment.url)||deployment.target!=='preview') throw Error('Unexpected deployment identity.');
const {env}=JSON.parse(await readFile('apps/sentinel-lab/.vercel/playground-manual.private.json','utf8'));
const cases=process.argv.includes('--expect-revoked')?[{name:'revoked-previous-key',token:true,origin:'https://eidosworks-playground-20261014.pages.dev',status:502}]:[
  {name:'replacement-key',token:true,origin:'https://eidosworks-playground-20261014.pages.dev',status:200},
  {name:'missing-owner-token',token:false,origin:'https://eidosworks-playground-20261014.pages.dev',status:401},
  {name:'wrong-site-origin',token:true,origin:'https://attacker.test',status:403},
];
const results=[];
for(const item of cases){
  const headers=['x-eidos-site-origin: '+item.origin,...(item.token?['x-eidos-platform-token: '+env.EIDOS_PLATFORM_TOKEN]:[])].join('\n')+'\n';
  const r=spawnSync(process.execPath,[args.cli,'curl','/api/playground-provider-check','--deployment',deployment.url,'--scope','team_TEsnMvKU6SdbYjVhws6Ttp8H','--','--header','@-','--silent','--show-error','--write-out','\n%{http_code}'],{cwd:'apps/sentinel-lab',encoding:'utf8',timeout:60000,input:headers,env:{...process.env,VERCEL_PROJECT_ID:'prj_5aL0COF1DONfPKI4fv25NitAdkOE',VERCEL_ORG_ID:'team_TEsnMvKU6SdbYjVhws6Ttp8H'}});
  assert.equal(r.status,0,'Protected CLI request failed; private output suppressed.');
  const lines=r.stdout.trim().split(/\r?\n/),status=Number(lines.pop());
  let value;try{value=JSON.parse(lines.join('\n'));}catch{throw Error('Unexpected response; raw output suppressed.');}
  const safe={name:item.name,status,expectedStatus:item.status,authenticated:value.authenticated===true,
    providerStatus:Number.isInteger(value.providerStatus)?value.providerStatus:null,
    providerRequestId:typeof value.providerRequestId==='string'&&/^[!-~]{1,200}$/.test(value.providerRequestId)?value.providerRequestId:null,
    models:{'gpt-6-astra':value.models?.['gpt-6-astra']===true,'gpt-6.1-sol':value.models?.['gpt-6.1-sol']===true},
    sourceCommit:/^[a-f0-9]{40}$/.test(value.sourceCommit||'')?value.sourceCommit:null,
    bundleHash:/^[a-f0-9]{64}$/.test(value.bundleHash||'')?value.bundleHash:null,passed:false};
  safe.passed=status===item.status;
  if(item.name==='replacement-key')safe.passed=safe.passed&&safe.authenticated&&safe.models['gpt-6-astra']&&safe.models['gpt-6.1-sol']&&safe.sourceCommit===deployment.sourceCommit&&safe.bundleHash===deployment.bundleHash;
  if(item.name==='revoked-previous-key')safe.passed=safe.passed&&safe.providerStatus===401&&!safe.authenticated;
  results.push(safe);
  if(!safe.passed)break;
}
const receipt={timestampUtc:new Date().toISOString(),deploymentId:deployment.id,deploymentUrl:deployment.url,results,
  generationRequests:0,countRequests:0,secretValuesRead:false,privateHeaders:'stdin',passed:results.every(x=>x.passed)};
await mkdir(out,{recursive:true});await writeFile(path.join(out,process.argv.includes('--expect-revoked')?'old-key-revocation.json':'replacement-authentication.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));if(!receipt.passed)process.exitCode=1;
