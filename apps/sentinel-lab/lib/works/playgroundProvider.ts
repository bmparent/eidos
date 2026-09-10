import {readText} from './vendor/functions/_shared/platform/core';
import type {AIProvider,ProviderResult,Usage} from './vendor/functions/_shared/platform/playgroundAI';
// Constructing the adapter performs no network activity. Only the admitted request calls it.
export function playgroundProvider(key:string|undefined):AIProvider|undefined {
 if(!key)return undefined;
 return async input=>{
  const image=input.kind==='image';
  const newer=/^gpt-(?:6|5\.[6-9])/.test(input.model);
  const payload=image?{model:input.model,prompt:input.content,n:1,quality:'low',size:'1024x1024',output_format:'webp',output_compression:80}:
  {model:input.model,store:false,input:[{role:'developer',content:[{type:'input_text',text:input.instructions,...(newer?{prompt_cache_breakpoint:{mode:'explicit'}}:{})}]},{role:'user',content:input.content}],text:{format:{type:'json_schema',name:'playground_edit',strict:true,schema:input.schema}},max_output_tokens:input.maxOutputTokens,prompt_cache_key:input.cacheKey,...(newer?{prompt_cache_options:{mode:'explicit',ttl:'30m'}}:{})};
  const response=await fetch('https://api.openai.com/v1/'+(image?'images/generations':'responses'),{method:'POST',headers:{authorization:'Bearer '+key,'content-type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(20000),redirect:'error'});
  // Never log response bodies, prompts or authorization. No automatic retry for any status.
  const raw=await readText(response,2_000_000);if(!response.ok)throw Error('Provider request did not complete.');const data=JSON.parse(raw);
  const requestId=response.headers.get('x-request-id')||data.id||'';
  const u=data.usage;let usage:Usage|undefined;
  if(u)usage={input:u.input_tokens,cachedInput:u.input_tokens_details?.cached_tokens??0,cacheWrite:u.input_tokens_details?.cache_write_tokens??0,output:image?0:u.output_tokens,reasoning:image?0:u.output_tokens_details?.reasoning_tokens??0,imageInput:image?u.input_tokens_details?.image_tokens??0:0,imageOutput:image?u.output_tokens:0,cachedImageInput:0};
  if(image){const bytes=data.data?.[0]?.b64_json;return {status:typeof bytes==='string'?'completed':'incomplete',image:typeof bytes==='string'?'data:image/webp;base64,'+bytes:undefined,usage,requestId,model:input.model};}
  const messages=(Array.isArray(data.output)?data.output:[]).flatMap((m:{type:string;content?:{type:string;text?:string}[]})=>m.type==='message'&&Array.isArray(m.content)?m.content:[]);
  const refused=messages.some((m:{type:string})=>m.type==='refusal');let proposal:unknown;
  try{proposal=JSON.parse(messages.filter((m:{type:string})=>m.type==='output_text').map((m:{text?:string})=>m.text||'').join(''));}catch{/* invalid/truncated output is charged and rejected by admission */}
  return {status:refused?'refused':data.status==='completed'?'completed':'incomplete',proposal,usage,requestId,model:data.model||input.model} as ProviderResult;
 };
}
