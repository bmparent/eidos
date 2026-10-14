/** Standalone reproducible numerical parity receipt; test-only replay. */
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {SentinelPort,ENGINE_PROVENANCE} from './brain.mjs';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const bytes = readFileSync(new URL('./brain-reference.json',import.meta.url));
const reference = JSON.parse(bytes), cases = [];
let comparisons=0,maxAbsolute=0,maxRelative=0,failures=[];
for (const fixture of reference.cases) {
  const detector=new SentinelPort(); let localMax=0;
  for(let i=0;i<fixture.frames.length;i++) {
    const actual=detector.step(fixture.frames[i]),expected=fixture.expected[i];
    for(const [key,want] of Object.entries(expected)) {
      comparisons++;
      if(typeof want==='number') {
        const error=Math.abs(actual[key]-want), relative=error/Math.max(1,Math.abs(want));
        maxAbsolute=Math.max(maxAbsolute,error);maxRelative=Math.max(maxRelative,relative);localMax=Math.max(localMax,error);
        if(!Number.isFinite(actual[key])||error>1e-7+Math.abs(want)*1e-8)failures.push(`${fixture.id}[${i}].${key}`);
      } else if(actual[key]!==want)failures.push(`${fixture.id}[${i}].${key}`);
    }
  }
  cases.push({id:fixture.id,provenance:fixture.provenance,frames:fixture.frames.length,max_absolute_error:localMax,source_sha256:fixture.source_sha256??null});
}
const receipt={schema:'eidos.sentinel.js-parity-receipt.v1',verified_at:new Date().toISOString(),
  method:'Recorded and test-only frames passed through canonical Python reference and JavaScript port, in sequence',
  numerical_tolerance:'1e-7 absolute + 1e-8 * abs(reference); exact status/string/boolean agreement',
  status:failures.length?'FAIL':'PASS',frames:cases.reduce((sum,c)=>sum+c.frames,0),field_comparisons:comparisons,
  max_absolute_error:maxAbsolute,max_relative_error_denominator_at_least_one:maxRelative,failures,cases,
  reference_sha256:sha(bytes),module_sha256:sha(readFileSync(new URL('./brain.mjs',import.meta.url))),
  canonical_source_hashes:ENGINE_PROVENANCE.canonical_source_hashes,
  limits:['One-feature,32-reservoir,warmup12,rolling8,seed42,cadence-neutral profile only',
    'Numerical equivalence for these inputs does not establish security detection quality or forecast skill',
    'Recorded-data parity replay does not create or score live-issued forecasts']};
writeFileSync(new URL('./brain-parity.json',import.meta.url),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt,null,2));
if(failures.length)process.exitCode=1;
