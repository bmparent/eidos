import test from 'node:test';
import assert from 'node:assert/strict';
import {playgroundProvider} from '../lib/works/playgroundProvider';
import {operationSchema} from '../lib/works/vendor/src/playground/aiOperations';

test('Playground adapter sends one bounded strict-schema request and preserves refusal/usage',async()=>{
 assert.equal(playgroundProvider(undefined),undefined);
 const prior=globalThis.fetch;let calls=0;
 globalThis.fetch=async(_url,init)=>{
  calls++;const input=JSON.parse(String(init?.body));assert.equal(input.store,false);assert.equal(input.text.format.strict,true);assert.equal(input.max_output_tokens,100);assert.equal(init?.redirect,'error');
  return Response.json({status:'completed',model:'fixture-model',usage:{input_tokens:12,output_tokens:4},output:[{type:'message',content:[{type:'refusal',refusal:'Fixture refusal'}]}]});
 };
 try{
  const provider=playgroundProvider('fixture-key')!;
  const input={kind:'text' as const,model:'fixture-model',content:'Fictional copy',instructions:'Fixture',maxOutputTokens:100,cacheKey:'fixture',schema:operationSchema};
  const result=await provider(input);assert.equal(result.status,'refused');assert.equal(result.usage?.input,12);assert.equal(calls,1);
  globalThis.fetch=async()=>{calls++;throw Error('Ambiguous network outcome');};
  await assert.rejects(()=>provider(input));assert.equal(calls,2);
 }finally{globalThis.fetch=prior;}
});
