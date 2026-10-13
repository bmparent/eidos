import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {vercelApi} from '../scripts/vercel-preview-client.mjs';

test('Vercel JSON writes declare the content type and keep payloads on stdin',()=>{
  const root=mkdtempSync(path.join(tmpdir(),'eidos-vercel-client-'));
  try {
    const cli=path.join(root,'mock-cli.mjs');
    writeFileSync(cli,"let input=''; for await (const chunk of process.stdin) input+=chunk; console.log(JSON.stringify({args:process.argv.slice(2),input}));");
    const payload=[{key:'SAFE_TEST_CONFIG',value:'fixture-only',target:['preview']}];
    const result=vercelApi(cli,'/fixture','POST',payload);
    assert.deepEqual(result.args,['api','/fixture','--method','POST','--raw','--non-interactive','--header','Content-Type: application/json','--input','-']);
    assert.deepEqual(JSON.parse(result.input),payload);
    assert.ok(!result.args.some(arg=>arg.includes('fixture-only')));
    const read=vercelApi(cli,'/fixture');
    assert.ok(!read.args.includes('--input')); assert.equal(read.input,'');
  } finally {rmSync(root,{recursive:true,force:true});}
});

test('Vercel request failures report a code without printing response secrets',()=>{
  const root=mkdtempSync(path.join(tmpdir(),'eidos-vercel-client-'));
  try {
    const cli=path.join(root,'mock-cli.mjs');
    writeFileSync(cli,"console.log(JSON.stringify({error:{code:'fixture_rejection',message:'private-fixture-value'}})); process.exitCode=1;");
    assert.throws(()=>vercelApi(cli,'/fixture'),error=>error.message==='Vercel API rejected the request: fixture_rejection');
  } finally {rmSync(root,{recursive:true,force:true});}
});
