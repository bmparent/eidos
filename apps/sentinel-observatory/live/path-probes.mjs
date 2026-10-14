/** Manual local ICMP/route probes with fixed command templates, no shell. */
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {lookup} from 'node:dns/promises';
import {isIP} from 'node:net';
import {normalizeTarget} from './collector.mjs';
const execute=promisify(execFile);
const safeIp=value=>{try{if(!isIP(value))return null;const t=normalizeTarget(value);return t.kind==='hostname'?null:t.hostname;}catch{return null;}};

export async function runPathProbe(input,kind,{exec=execute,resolver=lookup,platform=process.platform}={}) {
  if(!['ping','trace'].includes(kind))throw new Error('unsupported_path_probe');
  const target=normalizeTarget(input);let address=target.hostname;
  if(target.kind==='hostname'){
    let timer;
    try{
      const rows=await Promise.race([resolver(target.hostname,{all:true,verbatim:true}),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('dns_timeout')),4000);})]);
      if(!rows.length||rows.length>32||rows.some(row=>!safeIp(row.address)))throw new Error('non_public_dns');address=rows.find(row=>isIP(row.address)===4)?.address ?? rows[0].address;
    }finally{clearTimeout(timer);}
  }
  const win=platform==='win32',ipv6=isIP(address)===6;
  const command=kind==='ping'?'ping':win?'tracert':'traceroute';
  const args=kind==='ping'?win?[...(ipv6?['-6']:[]),'-n','3','-w','1000',address]:[...(ipv6?['-6']:[]),'-n','-c','3','-W','1',address]:win?['-d','-h','8','-w','500',address]:[...(ipv6?['-6']:[]),'-n','-m','8','-q','1','-w','0.5',address];
  const observed_at=new Date().toISOString();let output='';
  try{output=(await exec(command,args,{timeout:12000,maxBuffer:32768,windowsHide:true,encoding:'utf8'})).stdout ?? '';}
  catch(error){if(error.code==='ENOENT')return {state:'unavailable',reason:'system_command_not_installed',kind,observed_at};output=typeof error.stdout==='string'?error.stdout:'';if(!output)return {state:'unavailable',reason:error.killed?'command_deadline':'no_usable_command_result',kind,observed_at};}
  if(kind==='ping'){
    const timings=[...output.matchAll(/time[=<]\s*(\d+(?:\.\d+)?)\s*ms/gi)].map(x=>Number(x[1]));
    const match=output.match(/(\d+(?:\.\d+)?)%\s*(?:packet\s+)?loss/i);
    const loss=match?Math.min(1,Number(match[1])/100):null;
    if(!timings.length&&loss===null)return {state:'unavailable',reason:'unsupported_command_output',kind,observed_at};
    return {state:'ok',kind,observed_at,rtt_ms:timings.length?timings.reduce((a,b)=>a+b,0)/timings.length:null,loss_ratio:loss,samples:3,
      scope:'Three ICMP probes from this companion; missing replies do not establish a target outage'};
  }
  const hops=[];
  for(const line of output.split(/\r?\n/)){
    const hop=line.match(/^\s*(\d{1,2})\s+/);if(!hop||Number(hop[1])>8)continue;
    const addresses=[...new Set(line.split(/\s+/).map(token=>safeIp(token.replace(/^[([]|[)\],]$/g,''))).filter(Boolean))];
    hops.push({hop:Number(hop[1]),addresses,responded:!/^\s*\d+\s+\*\s*\*?\s*\*?\s*$/.test(line)});
  }
  return hops.length?{state:'ok',kind,observed_at,hops,scope:'At most eight hops; private router addresses omitted; ICMP filtering can hide hops'}:{state:'unavailable',reason:'no_parsable_hops',kind,observed_at};
}
