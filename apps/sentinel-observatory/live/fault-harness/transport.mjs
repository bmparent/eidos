/** TEST ONLY. No production module may import this loopback adapter. */
import * as http from 'node:http';
import * as https from 'node:https';
import * as net from 'node:net';
import {EventEmitter} from 'node:events';
import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';

const cert=readFileSync(new URL('fixtures/test-only-cert.pem',import.meta.url));
const key=readFileSync(new URL('fixtures/test-only-key.pem',import.meta.url));
const listen=server=>new Promise(resolve=>server.listen(0,'127.0.0.1',()=>resolve(server.address().port)));
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));

export async function createFaultTransport() {
  const sockets=new Set(),timers=new Set();let dynamic=0;
  const later=(work,ms)=>{const timer=setTimeout(()=>{timers.delete(timer);work();},ms);timers.add(timer);};
  function endpoint(req,res) {
    const mode=req.url.slice(1),head=req.method==='HEAD';
    res.setHeader('Content-Type','text/plain');
    if(['headers_timeout','https_headers_timeout','tls_partial_headers','cancel_headers','get405_fail'].includes(mode)&&(!head||mode!=='get405_fail'))return;
    if(mode==='headers_disconnect'){req.socket.destroy();return;}
    if(mode==='headers_delay'){later(()=>{if(!res.destroyed)res.end('delayed fixture');},15);return;}
    if(mode==='page_interrupted'&&!head){res.setHeader('Content-Length','10000');res.write('prefix');later(()=>res.socket?.destroy(),10);return;}
    if(mode==='page_unsupported')res.setHeader('Content-Type','image/png');
    const status=/^http(\d{3})$/.exec(mode);if(status)res.statusCode=Number(status[1]);
    if(['head405','get405_ok','get405_fail','get_redirect'].includes(mode)&&head)res.statusCode=405;
    if(['head501','get501_ok'].includes(mode)&&head)res.statusCode=501;
    const redirects={redirect_ok:'https://example.org/normal',redirect_final503:'https://example.org/http503',redirect_loop:'https://example.com/redirect_loop',
      redirect_limit:'https://example.org/redirect_limit',redirect_later_failure:'https://example.org/headers_timeout',redirect_not_followed:'https://example.org/normal',
      redirect_private:'http://127.0.0.1/',redirect_metadata:'http://169.254.169.254/',redirect_recursive:'https://collector.example/',
      page_redirect:'https://example.org/page_complete'};
    if(redirects[mode]||mode==='get_redirect'&&!head){res.statusCode=302;res.setHeader('Location',redirects[mode] ?? 'https://example.org/normal');}
    const body=mode==='page_prefix'?'x'.repeat(70000):mode==='page_dynamic'?`dynamic fixture ${++dynamic}`:mode==='zero_body'?'':'Sentinel controlled fixture';
    res.end(body);
  }
  const plain=http.createServer(endpoint),secure=https.createServer({key,cert},endpoint),blackhole=net.createServer();
  secure.on('tlsClientError',()=>{});
  for(const server of [plain,secure,blackhole])server.on('connection',socket=>{sockets.add(socket);socket.on('close',()=>sockets.delete(socket));});
  const ports={plain:await listen(plain),secure:await listen(secure),blackhole:await listen(blackhole)};
  const refused=net.createServer();ports.refused=await listen(refused);await new Promise(resolve=>refused.close(resolve));
  const trace=[];
  function factory(options,callback) {
    // Assert the production path supplied numeric public pinning and the
    // original Host/SNI/certificate verifier before mapping into the fixture.
    assert.ok(net.isIP(options.hostname));assert.ok(!net.isIP(options.headers.Host));
    assert.equal(options.rejectUnauthorized,true);assert.equal(options.agent,false);
    assert.ok([80,443].includes(options.port));assert.ok(!options.headers.Authorization);
    const mode=options.path.slice(1);trace.push({address:options.hostname,method:options.method,host:options.headers.Host,servername:options.servername,validation:true,mode});
    if(mode==='tcp_timeout'||mode==='alternate_timeout'&&options.hostname==='1.0.0.1'){
      const req=new EventEmitter();req.end=()=>{};req.destroy=error=>{if(error)queueMicrotask(()=>req.emit('error',error));return req;};return req;
    }
    if(mode==='tls_cleanup'){
      let alive=true;const req=new EventEmitter(),socket=new EventEmitter();
      Object.assign(socket,{remoteAddress:options.hostname,authorized:true,bytesRead:40,bytesWritten:60,getProtocol:()=>alive?'TLSv1.3':null,
        getPeerCertificate:()=>alive?{valid_from:'Oct 01 00:00:00 2026 GMT',valid_to:'Oct 01 00:00:00 2027 GMT',fingerprint256:'ab'.repeat(32)}:{}});
      req.destroy=error=>{alive=false;if(error)queueMicrotask(()=>req.emit('error',error));return req;};
      req.end=()=>queueMicrotask(()=>{req.emit('socket',socket);socket.emit('connect');socket.emit('secureConnect');
        const res=Object.assign(new EventEmitter(),{statusCode:200,headers:{'content-type':'text/plain'},destroy(){alive=false;return this;}});callback(res);});
      return req;
    }
    const useRefused=['tcp_refused','all_refused'].includes(mode)||['alternate_refused','bounded_answers'].includes(mode)&&options.hostname!=='1.1.1.1';
    const tls=options.protocol==='https:',port=useRefused?ports.refused:mode==='tls_timeout'?ports.blackhole:tls?ports.secure:ports.plain;
    return (tls?https:http).request({...options,hostname:'127.0.0.1',family:4,port,
      // Only the test adapter trusts this public fixture certificate. The
      // original checkServerIdentity and rejectUnauthorized remain in force.
      ...(tls&&mode!=='tls_untrusted'?{ca:cert}:{})},callback);
  }
  const normalLookup=async()=>[{address:'1.1.1.1',family:4}];
  function optionsFor(fault) {
    const options={lookup:normalLookup,requestFactory:factory,timeoutMs:1000,blockedHosts:['collector.example']};
    // Allow real TLS setup on loaded Windows hosts. Deliberately slow fixtures
    // still share this one measured deadline; none gets an insecure shortcut.
    if(['dns_timeout','tcp_timeout','tls_timeout','tls_partial_headers','headers_timeout','https_headers_timeout','get405_fail'].includes(fault))options.timeoutMs=1000;
    if(fault==='dns_missing')options.lookup=async()=>{throw Object.assign(new Error('Controlled DNS miss'),{code:'ENOTFOUND'});};
    if(['dns_timeout','dns_cancel'].includes(fault))options.lookup=()=>new Promise(()=>{});
    if(fault==='dns_empty')options.lookup=async()=>[];
    if(fault==='dns_delay')options.lookup=async()=>{await wait(15);return normalLookup();};
    if(['dns_rotation','normal_ipv6'].includes(fault))options.lookup=async()=>[{address:'2606:4700:4700::1111',family:6}];
    const unsafe={mixed_ipv4:'127.0.0.1',mixed_ipv6:'::1',mapped_private:'::ffff:127.0.0.1'};
    if(unsafe[fault])options.lookup=async()=>[{address:'1.1.1.1'},{address:unsafe[fault]}];
    if(['alternate_refused','alternate_timeout','all_refused','bounded_answers'].includes(fault))options.lookup=async()=>['1.0.0.1','1.0.0.2',...(fault==='bounded_answers'?['1.0.0.3']:[]),'1.1.1.1'].map(address=>({address,family:4}));
    if(fault.startsWith('redirect_')||fault==='page_redirect'||fault==='get_redirect')options.follow_redirects=fault!=='redirect_not_followed';
    if(fault.startsWith('get'))options.get_fallback=true;
    if(fault.startsWith('page_'))options.page_check=true;
    if(['dns_cancel','cancel_headers'].includes(fault)){
      const controller=new AbortController();options.signal=controller.signal;later(()=>controller.abort(),25);
    }
    return options;
  }
  return {factory,optionsFor,trace,ports,async close(){for(const timer of timers)clearTimeout(timer);for(const socket of sockets)socket.destroy();await Promise.all([plain,secure,blackhole].map(server=>new Promise(resolve=>server.close(resolve))));}};
}
