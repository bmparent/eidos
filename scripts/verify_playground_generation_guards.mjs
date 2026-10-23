/** Hosted negative/replay checks after a completed generation; no new action IDs. */
import { readFile,writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if(!args.cli||!args.out)throw Error('Supply --cli and versioned --out.');
const out=path.resolve(args.out),root=path.resolve('artifacts/playground-authenticated-20261015');
if(!out.startsWith(root+path.sep))throw Error('Unexpected evidence directory.');
const original=await readFile(path.join(out,'generation-http-receipt.json')),receipt=JSON.parse(original);
if(!receipt.passed||receipt.provider.state!=='generated')throw Error('Never replay an unknown generation. Read the durable ledger first.');
const deployment=JSON.parse(await readFile(path.join(out,'generation-deployment.json'),'utf8'));
if(deployment.target!=='preview'||!/^https:\/\/eidos-sentinel-[a-z0-9]+-1brentbm-1876s-projects\.vercel\.app$/.test(deployment.url))throw Error('Unexpected protected Preview.');
const {env}=JSON.parse(await readFile('apps/sentinel-lab/.vercel/playground-manual.private.json','utf8'));
const checks=[];
for(const [name,includeToken,origin,expected] of [['missing_owner_token',false,'https://eidosworks-playground-20261014.pages.dev',401],['wrong_origin',true,'https://invalid.example.test',403],['settled_replay_1',true,'https://eidosworks-playground-20261014.pages.dev',409],['settled_replay_2',true,'https://eidosworks-playground-20261014.pages.dev',409]]){
  const headers=['x-eidos-site-origin: '+origin,'content-type: application/json'];if(includeToken)headers.push('x-eidos-platform-token: '+env.EIDOS_PLATFORM_TOKEN);
  const result=spawnSync(process.execPath,[args.cli,'curl','/api/playground-provider-generation','--deployment',deployment.url,'--scope','team_TEsnMvKU6SdbYjVhws6Ttp8H','--','--request','POST','--header','@-','--data-binary','{}','--silent','--show-error','--write-out','\n%{http_code}'],{
    cwd:'apps/sentinel-lab',encoding:'utf8',timeout:45000,input:headers.join('\n')+'\n',env:{...process.env,VERCEL_PROJECT_ID:'prj_5aL0COF1DONfPKI4fv25NitAdkOE',VERCEL_ORG_ID:'team_TEsnMvKU6SdbYjVhws6Ttp8H'},
  });
  const lines=(result.stdout||'').trim().split(/\r?\n/),status=Number(lines.pop());let body={};try{body=JSON.parse(lines.join('\n'));}catch{/* No raw error output. */}
  checks.push({name,status,expected,returnedGenerationRequests:body.generationRequests??null,passed:result.status===0&&status===expected&&(!name.startsWith('settled_replay')||body.generationRequests===0)});
}
const local=spawnSync(process.execPath,['scripts/submit_playground_generation_probe.mjs','--cli='+args.cli,'--receipt='+path.join(out,'generation-deployment.json'),'--out='+out],{encoding:'utf8',timeout:10000});
const unchanged=createHash('sha256').update(original).digest('hex')===createHash('sha256').update(await readFile(path.join(out,'generation-http-receipt.json'))).digest('hex');
const proof={timestampUtc:new Date().toISOString(),deploymentId:deployment.id,checks,localNoResubmitGuard:local.status===1&&String(local.stderr).includes('generation was already started'),originalReceiptUnchanged:unchanged,providerKeyReadLocally:false,secretValuesPrinted:false};
proof.passed=checks.every(c=>c.passed)&&proof.localNoResubmitGuard&&unchanged;
await writeFile(path.join(out,'generation-hosted-guards.json'),JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify(proof));if(!proof.passed)process.exitCode=1;
