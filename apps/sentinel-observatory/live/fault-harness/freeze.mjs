import {readFileSync,writeFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
const hash=path=>createHash('sha256').update(readFileSync(new URL(path,import.meta.url))).digest('hex');
const lockPath=new URL('registry-lock.json',import.meta.url),hashes=Object.fromEntries(['scenarios.json','protocol.json'].map(path=>[path,hash(path)]));
if(existsSync(lockPath)){
  const prior=JSON.parse(readFileSync(lockPath));if(JSON.stringify(prior.hashes)!==JSON.stringify(hashes))throw new Error('Frozen protocol changed: create a new version, preserve previous runs');
}else writeFileSync(lockPath,JSON.stringify({schema:'observatory-registry-lock-v1',frozen_at:new Date().toISOString(),hashes},null,2)+'\n');
console.log(JSON.stringify({frozen:true,hashes}));
