import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {spawn} from 'node:child_process';
const root=resolve(import.meta.dirname,'../../..'),base='http://127.0.0.1:3210';
const privateRoot=resolve(root,'artifacts/sentinel-guided-private/collector');
const output=resolve(root,'artifacts/sentinel-guided-20260908/collector');mkdirSync(output,{recursive:true});
const keys=JSON.parse(readFileSync(resolve(root,'artifacts/sentinel-guided-private/local-access.json')));
const api=async(path,body)=>{const r=await fetch(`${base}/api/lab/v1/${path}`,{method:body?'POST':'GET',headers:{Authorization:`Bearer ${keys.bob}`,'Content-Type':'application/json'},...(body?{body:JSON.stringify(body)}:{})});const d=await r.json();if(!r.ok)throw Error(`API ${r.status}`);return d;};
const receipt={startedAt:new Date().toISOString(),version:'0.160.0',input:'three explicitly synthetic OTLP gauge points',serviceInstalled:false,unrelatedCollection:false};
let collector,logs='',credential='';
try{
 const {monitor,key}=await api('monitors',{name:'Official collector synthetic forwarding',target:'validation.latency',unit:'ms',synthetic:true,autoProcess:false});credential=key;receipt.sourceId=monitor.id;
 const config=readFileSync(resolve(root,'connectors/sentinel/otel-collector.yaml'),'utf8').replaceAll('127.0.0.1:4317','127.0.0.1:49317').replaceAll('127.0.0.1:4318','127.0.0.1:49318').replace('directory: ./sentinel-collector-state',`directory: '${resolve(privateRoot,'queue').replaceAll('\\','/')}'`);
 writeFileSync(resolve(privateRoot,'validation.yaml'),config);
 collector=spawn(process.env.EIDOS_COLLECTOR_BIN || resolve(privateRoot,'otelcol-contrib.exe'),['--config',resolve(privateRoot,'validation.yaml')],{cwd:root,windowsHide:true,env:{PATH:process.env.PATH,SystemRoot:process.env.SystemRoot,TEMP:process.env.TEMP,EIDOS_INGEST_URL:`${base}/api/lab/v1/telemetry/${monitor.id}`,EIDOS_INGEST_KEY:key},stdio:['ignore','pipe','pipe']});
 collector.stdout.on('data',d=>logs+=d);collector.stderr.on('data',d=>logs+=d);collector.on('error',e=>logs+=e.message);
 for(let i=0;i<20&&!logs.includes('Everything is ready');i++)await new Promise(r=>setTimeout(r,500));
 const payload={resourceMetrics:[{resource:{attributes:[{key:'service.name',value:{stringValue:'eidos-authorized-connector-validation'}}]},scopeMetrics:[{metrics:[{name:'validation.latency',unit:'ms',gauge:{dataPoints:[0,1,2].map(i=>({timeUnixNano:String((BigInt(Date.UTC(2026,7,5))+BigInt(i*1000))*1000000n),asDouble:40+i}))}}]}]}]};
 const response=await fetch('http://127.0.0.1:49318/v1/metrics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!response.ok)throw Error(`Collector receive HTTP ${response.status}`);
 let state;for(let i=0;i<40;i++){state=await api(`monitors/${monitor.id}`);if(state.lastOffset===3)break;await new Promise(r=>setTimeout(r,2000));}
 if(state.lastOffset!==3||state.events.some(e=>!e.synthetic))throw Error('Collector did not forward three accounted synthetic points');
 receipt.accepted=state.lastOffset;receipt.eventValues=state.events.map(e=>e.value).sort();receipt.status='passed';
 await api(`monitors/${monitor.id}/revoke`,{});receipt.keyRevoked=true;
}catch(e){receipt.status='failed';receipt.failure=e.message;process.exitCode=1;}finally{
 if(collector){collector.kill();await new Promise(r=>setTimeout(r,500));}
 receipt.finishedAt=new Date().toISOString();writeFileSync(resolve(output,'collector.log'),credential?logs.replaceAll(credential,'[REDACTED]'):logs);writeFileSync(resolve(output,'receipt.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));
}
