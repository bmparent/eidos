import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
const base='https://eidos-sentinel-lab.vercel.app';
const job='rd-8a14916b3ea7-696b7427';
const results=[];
for(const path of [`/api/experiments/${job}`,`/api/experiments/${job}/artifacts/artifact_verification.json`,`/api/experiments/${job}/artifacts/metrics.json`]) {
  const response=await fetch(base+path,{signal:AbortSignal.timeout(30000)});
  results.push({path,status:response.status});
  assert.equal(response.status,401);
}
const value={checkedAt:new Date().toISOString(),authenticationEnforced:true,results};
writeFileSync(new URL('public-boundary-check.json',import.meta.url),JSON.stringify(value,null,2)+'\n');
console.log(JSON.stringify(value));
