import test from 'node:test';
import assert from 'node:assert/strict';
import {runPathProbe} from './path-probes.mjs';
import {createCompanion} from './node-companion.mjs';

test('path probes use a pinned public numeric destination and a fixed command without a shell',async()=>{
  const result=await runPathProbe('https://example.com','ping',{resolver:async()=>[{address:'1.1.1.1'}],platform:'linux',exec:async(command,args,options)=>{
    assert.equal(command,'ping');assert.equal(args.at(-1),'1.1.1.1');assert.equal(options.shell,undefined);
    return {stdout:'64 bytes from 1.1.1.1: time=10.5 ms\n64 bytes from 1.1.1.1: time=12.5 ms\n3 packets transmitted, 2 received, 33.3333% packet loss'};
  }});
  assert.equal(result.rtt_ms,11.5);assert.ok(result.loss_ratio>0.33&&result.loss_ratio<0.34);
});
test('missing ICMP tools and filtered replies remain explicit rather than healthy zero latency',async()=>{
  const absent=await runPathProbe('1.1.1.1','ping',{exec:async()=>{throw Object.assign(new Error(),{code:'ENOENT'});}});
  assert.equal(absent.state,'unavailable');
  const filtered=await runPathProbe('1.1.1.1','ping',{exec:async()=>({stdout:'3 packets transmitted, 0 received, 100% packet loss'})});
  assert.equal(filtered.loss_ratio,1);assert.equal(filtered.rtt_ms,null);
});
test('traces bound hop counts and omit private router addresses',async()=>{
  const result=await runPathProbe('1.1.1.1','trace',{exec:async()=>({stdout:'1 192.168.1.1 1.0 ms\n2 *\n3 1.1.1.1 12 ms\n9 8.8.8.8 19 ms'})});
  assert.equal(result.hops.length,3);assert.deepEqual(result.hops[0].addresses,[]);assert.deepEqual(result.hops[2].addresses,['1.1.1.1']);
});
test('local companion serves probe UI and rejects remote-host or cross-origin collection',async()=>{
  const {server,database}=createCompanion({port:0});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base='http://127.0.0.1:'+server.address().port;
  try{
    assert.equal((await fetch(base+'/target-panel.mjs')).status,200);
    const capabilities=await (await fetch(base+'/api/capabilities')).json();assert.equal(capabilities.target_probes.connected,true);
    assert.equal((await fetch(base+'/api/observe',{method:'POST',headers:{Origin:'https://attacker.example','Content-Type':'application/json'},body:'{"target":"1.1.1.1"}'})).status,403);
    assert.equal((await fetch(base+'/api/observe',{method:'POST',headers:{Origin:base,'Content-Type':'application/json'},body:'{"target":"127.0.0.1"}'})).status,400);
  }finally{await new Promise(resolve=>server.close(resolve));database.close();}
});
