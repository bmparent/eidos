/** Node 24+ companion: all endpoint probes, durable analysis and optional path tools. */
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {homedir} from 'node:os';
import {pathToFileURL} from 'node:url';
import {handleObservation,normalizeTarget} from './collector.mjs';
import {observeTarget} from './node-probes.mjs';
import {retainTargetObservation,targetHistory} from './target-store.mjs';
import {handleInterpretation,hostedInterpretationCapability} from './sol-interpreter.mjs';
import {runPathProbe} from './path-probes.mjs';
import {CompanionDatabase} from './sqlite-store.mjs';

export function createCompanion({database=new CompanionDatabase(),port=8767,pathProbe=runPathProbe,observer=observeTarget}={}) {
  const pending=new Map(),pathRates=new Map();let pathActive=false;
  const env={OBSERVATORY_DB:database,OBSERVATORY_TARGET_PROBES_ENABLED:'true',OPENAI_API_KEY:process.env.OPENAI_API_KEY};
  const json=(data,status=200)=>new Response(JSON.stringify(data),{status,headers:{'content-type':'application/json','cache-control':'no-store','x-content-type-options':'nosniff'}});
  const assets={'/':'web/index.html','/app.js':'web/app.js','/styles.css':'web/styles.css','/brain.mjs':'brain.mjs','/target-panel.mjs':'web/target-panel.mjs','/network-panel.mjs':'web/network-panel.mjs','/security-events.mjs':'security-events.mjs'};
  const server=createServer(async(incoming,outgoing)=>{
    try{
      const address=server.address(),actualPort=typeof address==='object'?address.port:port,origin=`http://127.0.0.1:${actualPort}`;
      if(incoming.headers.host!==`127.0.0.1:${actualPort}`){outgoing.writeHead(403);outgoing.end();return;}
      const url=new URL(incoming.url,origin);let response;
      if(incoming.method==='POST'){
        if(incoming.headers.origin!==origin){outgoing.writeHead(403);outgoing.end();return;}
        let size=0;const chunks=[];for await(const chunk of incoming){size+=chunk.length;if(size>1536){outgoing.writeHead(413);outgoing.end();return;}chunks.push(chunk);}
        const body=Buffer.concat(chunks).toString('utf8'),request=new Request(url,{method:'POST',headers:{Origin:origin,'Content-Type':incoming.headers['content-type'] ?? ''},body});
        if(url.pathname==='/api/observe'){
          response=await handleObservation(request,env,{targetCollect:async(target,options)=>{
            const result=await observer(target.url,options),path=pending.get(target.url);
            if(path&&Date.now()-Date.parse(path.observed_at)<60000){result.event.probes.path=path;pending.delete(target.url);}
            result.event=await retainTargetObservation(env,result.event);return result;
          }});
        }else if(url.pathname==='/api/interpret')response=await handleInterpretation(request,env);
        else if(url.pathname==='/api/path-probe'){
          const payload=JSON.parse(body);if(Object.keys(payload).sort().join(',')!=='kind,target'||!['ping','trace'].includes(payload.kind))throw new Error('invalid_path_request');
          const target=normalizeTarget(payload.target),now=Date.now();
          if(pathActive||now-(pathRates.get(target.url) ?? 0)<15000)response=json({message:'Wait for the current diagnostic or its cooldown.'},429);
          else{pathActive=true;pathRates.set(target.url,now);try{const result=await pathProbe(target.url,payload.kind);const previous=pending.get(target.url);
              const combined=result.kind==='ping'?{...previous,...result}:previous?.kind==='ping'?{...previous,hops:result.hops,trace_state:result.state,trace_observed_at:result.observed_at}:result;
              if(result.state==='ok')pending.set(target.url,combined);response=json({target_id:target.id,result});
            }finally{pathActive=false;}}
        }else response=json({error:'not_found'},404);
      }else if(incoming.method==='GET'){
        if(url.pathname==='/api/capabilities')response=json({mode:'local',direct_ip:true,interval_ms:15000,max_targets:4,hosted_interpretation:hostedInterpretationCapability(env),model:hostedInterpretationCapability(env),target_probes:{connected:true,path:'local_commands',profile:'target-metrics-v1'},security_sensor:{connected:false,reason:'Use retained target evidence; no network log sensor attached'}});
        else if(url.pathname==='/api/target-history')response=json(await targetHistory(env,url.searchParams.get('target_id')));
        else if(url.pathname==='/api/model/status')response=json({available:false,connected:false,reason:'Optional Python/Qwen companion remains a separate launcher.'});
        else if(assets[url.pathname]){const file=assets[url.pathname],type=file.endsWith('.css')?'text/css':file.endsWith('.html')?'text/html':'text/javascript';response=new Response(await readFile(new URL(file,import.meta.url)),{headers:{'content-type':type+'; charset=utf-8','cache-control':'no-store'}});}
        else response=json({error:'not_found'},404);
      }else response=json({error:'method_not_allowed'},405);
      outgoing.writeHead(response.status,Object.fromEntries(response.headers));outgoing.end(Buffer.from(await response.arrayBuffer()));
    }catch{outgoing.writeHead(400,{'content-type':'application/json','cache-control':'no-store'});outgoing.end(JSON.stringify({error:'bounded_request_failed'}));}
  });
  server.requestTimeout=20000;server.headersTimeout=5000;
  return {server,database};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const directory=join(homedir(),'.sentinel-observatory');mkdirSync(directory,{recursive:true});
  const database=new CompanionDatabase(join(directory,'target-evidence.sqlite')),companion=createCompanion({database});
  companion.server.listen(8767,'127.0.0.1',()=>process.stdout.write('Sentinel target companion: http://127.0.0.1:8767\nHistory is retained locally. Ctrl-C stops collection.\n'));
  process.on('SIGINT',()=>companion.server.close(()=>{database.close();process.exit(0);}));
}
