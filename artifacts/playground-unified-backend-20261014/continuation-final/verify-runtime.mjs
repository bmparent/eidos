// Read-only protected Preview check. Sensitive relay header is sent on stdin.
import { readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
const cli=process.argv[2],out=process.argv[3]||'artifacts/playground-unified-backend-20261014/continuation-final/runtime-flags.json';
if(!cli)throw Error('Supply the authenticated CLI entry.');
const deployed=JSON.parse(readFileSync('artifacts/playground-unified-backend-20261014/backend-preview.json','utf8'));
if(deployed.manualOnly!==true||deployed.deployment?.target!=='preview'||deployed.deployment.state!=='READY')throw Error('A READY manual Preview receipt is required.');
const url=deployed.deployment.url;
const endpoint=url+'/api/works/v1/api/public-config';
const unauthenticated=await fetch(endpoint,{redirect:'manual',signal:AbortSignal.timeout(30000)});
const isolated=JSON.parse(readFileSync('apps/sentinel-lab/.vercel/playground-manual.private.json','utf8')).env;
const token=isolated.EIDOS_PLATFORM_TOKEN;
if(typeof token!=='string'||token.length<32)throw Error('Existing private relay configuration is incomplete.');
const response=spawnSync(process.execPath,[cli,'curl',endpoint,'--deployment',url,'--scope','1brentbm-1876s-projects','--','--silent','--show-error','--max-time','30','--header','@-','--write-out','\n%{http_code}'],{encoding:'utf8',input:'x-eidos-platform-token: '+token+'\nx-eidos-site-origin: https://eidosworks-playground-20261014.pages.dev\nOrigin: https://eidosworks-playground-20261014.pages.dev\n',maxBuffer:128000,timeout:90000});
const raw=response.stdout||'',split=raw.lastIndexOf('\n');
let data;try{data=JSON.parse(split>=0?raw.slice(0,split):raw);}catch{data={};}
const status=split>=0?Number(raw.slice(split+1)):null;
const flags=Object.fromEntries(['communityReady','accountsReady','aiReady','playgroundAiReady','shopReady','localTest'].map(key=>[key,typeof data[key]==='boolean'?data[key]:null]));
const receipt={timestampUtc:new Date().toISOString(),deploymentId:deployed.deployment.id,url,expectedSource:deployed.backendSha,expectedBundle:deployed.bundleHash,unauthenticatedStatus:unauthenticated.status,authenticatedStatus:status,cliExitCode:response.status,flags,runtimeSourceRevision:data.runtimeSourceRevision||null,runtimeBundleHash:data.runtimeBundleHash||null,secretsPrinted:false,headerTransport:'stdin to native curl --header @-',providerCalls:0,scope:'This exact manual-only source stage; final candidate acceptance is separate'};
if(response.status!==0)receipt.diagnostic=String(response.stderr||'').replaceAll(token,'[REDACTED]').replace(/https?:\/\/[^\s]+/g,'[REDACTED_URL]').replace(/(?:authorization|Bearer|x-eidos-platform-token|x-vercel-protection-bypass|x-vercel-trusted-oidc-idp-token)[^\r\n]*/gi,'[REDACTED_HEADER]').replace(/[A-Za-z0-9_-]{32,}/g,'[REDACTED_LONG_VALUE]').slice(0,1200);
receipt.passed=response.status===0&&status===200&&Object.values(flags).every(value=>value===false)&&receipt.runtimeSourceRevision===deployed.backendSha&&receipt.runtimeBundleHash===deployed.bundleHash&&[302,307,401,403].includes(unauthenticated.status);
if(!receipt.passed&&typeof data.reason==='string')receipt.cliReason=data.reason;
writeFileSync(out,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
if(!receipt.passed)process.exitCode=1;
