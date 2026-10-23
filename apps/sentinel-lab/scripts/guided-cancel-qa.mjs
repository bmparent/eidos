import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'../../..'),base='http://127.0.0.1:3210';
const keys=JSON.parse(readFileSync(resolve(root,'artifacts/sentinel-guided-private/local-access.json')));
const out=resolve(root,'artifacts/sentinel-guided-20260908/cancellation');mkdirSync(out,{recursive:true});
const receipt={startedAt:new Date().toISOString(),checks:[]};
const check=(name,ok)=>{if(!ok)throw Error(name);receipt.checks.push({name,status:'passed'});};
async function api(path,data,key=crypto.randomUUID().replaceAll('-','_')) {
 const r=await fetch(`${base}/api/lab/v1/${path}`,{method:data===undefined?'GET':'POST',headers:{Authorization:`Bearer ${keys.alice}`,'Content-Type':'application/json','Idempotency-Key':key},...(data===undefined?{}:{body:JSON.stringify(data)})});
 return {status:r.status,body:await r.json()};
}
try {
 const runs=await api('runs');const previous=runs.body.find(r=>r.method.includes('Torch'));if(!previous)throw Error('Run CSV acceptance first');
 const intent=crypto.randomUUID().replaceAll('-','_');const input={mode:'forecast',target:'latency_ms',horizonSeconds:60,windowSeconds:60};
 const first=await api(`datasets/${previous.datasetId}/analyze`,input,intent);check('job accepted before cancellation',first.status===202);
 const replay=await api(`datasets/${previous.datasetId}/analyze`,input,intent);check('same intent returns same worker identity',replay.body.id===first.body.id);
 const changed=await api(`datasets/${previous.datasetId}/analyze`,{...input,horizonSeconds:120},intent);check('same intent with changed input fails',changed.status===409);
 const cancelled=await api(`jobs/${first.body.id}/cancel`,{});check('active job cancelled',cancelled.body.status==='cancelled');
 await new Promise(r=>setTimeout(r,5000));
 const after=await api(`jobs/${first.body.id}`);check('late worker cannot publish completed',after.body.status==='cancelled');
 check('cancelled job has no visible result',(await api(`runs/${first.body.id}`)).status===404);
 const next=await api(`datasets/${previous.datasetId}/analyze`,input);check('subsequent job accepted',next.status===202);receipt.cancelledJobId=first.body.id;receipt.recoveryJobId=next.body.id;
 let job=next.body;
 for(let i=0;i<90&&!['completed','failed','cancelled','expired'].includes(job.status);i++){
   if(job.status==='queued')await api(`jobs/${job.id}/resume`,{});await new Promise(r=>setTimeout(r,2000));job=(await api(`jobs/${job.id}`)).body;
 }
 check('capacity reclaimed and genuine next job completes',job.status==='completed');receipt.status='passed';
}catch(e){receipt.status='failed';receipt.failure=e.message;process.exitCode=1;}
finally{receipt.finishedAt=new Date().toISOString();writeFileSync(resolve(out,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));}
