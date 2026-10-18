/** Explicit local-only fault demo, isolated from real companion/hosted history. */
import {resolve} from 'node:path';
import {mkdirSync} from 'node:fs';
import {createCompanion} from '../node-companion.mjs';
import {CompanionDatabase} from '../sqlite-store.mjs';
import {observeTarget} from '../node-probes.mjs';
import {normalizeTarget} from '../collector.mjs';
import {normalizeTargetProbe} from '../target-relay.mjs';
import {createFaultTransport} from './transport.mjs';
import {registry} from './run.mjs';
const port=Number(process.env.SENTINEL_FIXTURE_PORT ?? 8788),dir=resolve('artifacts/proof_runs/2026-10-18/observatory-m1/preview');mkdirSync(dir,{recursive:true});
const database=new CompanionDatabase(resolve(dir,'synthetic-evidence.sqlite')),transport=await createFaultTransport();
// Do not use a configured real model key with synthetic demo evidence.
const originalKey=process.env.OPENAI_API_KEY;delete process.env.OPENAI_API_KEY;
const companion=createCompanion({database,port,decorateHtml:html=>html.replace('<main>','<main><p role="note" style="padding:16px;border:1px solid #efbc7f;color:#efbc7f">SYNTHETIC LOCAL FAULT HARNESS · no external endpoint requests. Try https://example.com/tls_partial_headers, http://example.com/alternate_refused, https://example.com/get405_ok or https://example.com/redirect_final503. Foreground collection only.</p>'),
  pathProbe:async()=>({state:'unavailable',reason:'fixture_demo_disables_path_commands'}),observer:async(input,options)=>{
    const target=normalizeTarget(input),fault=new URL(target.url).pathname.slice(1);
    if(!['example.com','example.org','wrong.example.com'].includes(target.hostname)||!registry.cases.some(x=>x.fault===fault))throw new Error('Choose a registered synthetic fixture; external probing is disabled.');
    const testOptions={...options,...transport.optionsFor(fault)},raw=await observeTarget(input,testOptions);
    return {...raw,event:{...normalizeTargetProbe(raw,target,testOptions),evidence_class:'synthetic_controlled'}};
  }});
if(originalKey!==undefined)process.env.OPENAI_API_KEY=originalKey;
companion.server.listen(port,'127.0.0.1',()=>console.log(`Synthetic Observatory fault preview: http://127.0.0.1:${port}`));
async function close(){await new Promise(resolve=>companion.server.close(resolve));await transport.close();database.close();}
process.once('SIGINT',()=>close().then(()=>process.exit()));process.once('SIGTERM',()=>close().then(()=>process.exit()));
