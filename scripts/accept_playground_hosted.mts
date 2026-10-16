/** Genuine hosted TEST account/delivery acceptance. No fake sessions, payments or provider transport. */
import { readFile, writeFile, mkdir, unlink } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { createInterface } from 'node:readline/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { adaptDatabase } from '../apps/sentinel-lab/lib/works/database';
import { grantPaidAiBalance, aiBalance, aiPolicyStatus } from '../apps/sentinel-lab/lib/works/vendor/functions/_shared/platform/playgroundAi';
import { ownerCountRequest } from '../apps/sentinel-lab/lib/works/playground-count-probe';
import { applyContextProposal } from '../apps/sentinel-lab/lib/works/project-delivery/src/playground/aiOperations';
import { upgradeProject } from '../apps/sentinel-lab/lib/works/project-delivery/src/playground/model';
import { createProjectSnapshot, buildFrozenPackage } from '../apps/sentinel-lab/lib/works/project-delivery/src/playground/purchaseSnapshot';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
const root=path.resolve('artifacts/playground-authenticated-20261015/release-continuation'), privateRoot=path.resolve('apps/sentinel-lab/.vercel');
const privateFile=path.join(privateRoot,'playground-test-sessions.private.json'), origin='https://eidosworks-playground-test-20261015.pages.dev';
const deployment=JSON.parse(await readFile(path.join(root,'test-backend-deployment.json'),'utf8'));
if(!args.cli||deployment.deployment.state!=='READY'||deployment.siteOrigin!==origin||deployment.mode!=='test'||deployment.production!==false)throw Error('Use the frozen READY isolated TEST deployment.');
const relay=JSON.parse(await readFile(path.join(privateRoot,'playground-manual.private.json'),'utf8')).env.EIDOS_PLATFORM_TOKEN;
if(relay.length<32)throw Error('Existing relay required.');
const sha=(data:string|Uint8Array)=>createHash('sha256').update(data).digest('hex');
const actors={business:{email:'brent@eidos-works.com',username:'brent_pg_test',action:'signup'},personal:{email:'1brent.bm@gmail.com',action:'signin'}};
let state:Record<string,any>={};try{state=JSON.parse(await readFile(privateFile,'utf8'));}catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}
await mkdir(root,{recursive:true});await mkdir(privateRoot,{recursive:true});
const saveState=()=>writeFile(privateFile,JSON.stringify(state),{mode:0o600});
async function privateInput() {
  // An interactive runner must disable both console and readline echo before the secret arrives.
  const terminal=process.stdin.isTTY,wasRaw=process.stdin.isRaw;
  if(terminal)process.stdin.setRawMode(true);
  const rl=createInterface({input:process.stdin,terminal:false});
  try {
    console.log('Private inbox token input ready; echo disabled.');
    const line=await rl.question('');
    const input=JSON.parse(line);if(!/^[a-f0-9]{64}$/.test(input.token))throw Error('Supply the one-time inbox token privately on stdin.');return input.token as string;
  } finally { rl.close();process.stdin.pause();if(terminal)process.stdin.setRawMode(Boolean(wasRaw)); }
}
async function testDatabase() {
  if(!args.configuration)throw Error('Supply the existing ignored approved TEST configuration.');
  const configuration=JSON.parse(await readFile(args.configuration,'utf8')).env;
  if(configuration.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!new URL(configuration.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation'))throw Error('Only the approved TEST database is allowed.');
  const require=createRequire(path.resolve('apps/sentinel-lab/package.json')),{createClient}=require('@libsql/client');
  const client=createClient({url:configuration.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL,authToken:configuration.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN});
  const env={EIDOS_DB:adaptDatabase(client),EIDOS_PLAYGROUND_AI_MODE:'test',EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID:'acct_1Rj03gKC8pRG5Tr9'};
  return {configuration,client,env};
}
async function call(route:string,body?:unknown,slot?:string,diagnostic=false) {
  if(!(diagnostic&&route==='/api/playground-provider-check')&&route!=='/api/public-config'&&!/^\/api\/(?:members|playground|shop)\/[a-z-]+(?:\?[^#]*)?$/.test(route))throw Error('Unexpected API route.');
  const id=randomBytes(12).toString('hex'), headerFile=path.join(privateRoot,id+'.headers.private'),bodyFile=path.join(privateRoot,id+'.body.private'),outputHeaders=path.join(privateRoot,id+'.response.private');
  const headers=['x-eidos-platform-token: '+relay,'x-eidos-site-origin: '+(diagnostic?'https://eidosworks-playground-20261014.pages.dev':origin),'origin: '+origin,'content-type: application/json',...(diagnostic&&args.phase!=='provider-metadata'?['x-eidos-bindings-only: true']:[])];
  const session=slot?state[slot]?.cookie:undefined;
  if(session){if(!/^__Host-eidos_session=[a-f0-9]{64}$/.test(session))throw Error('Invalid saved session shape.');headers.push('cookie: '+session);}
  await writeFile(headerFile,headers.join('\n')+'\n',{mode:0o600,flag:'wx'});
  if(body!==undefined)await writeFile(bodyFile,JSON.stringify(body),{mode:0o600,flag:'wx'});
  try {
    const command=[args.cli,'curl',(diagnostic?'':'/api/works/v1')+route,'--deployment',deployment.deployment.url,'--scope','team_TEsnMvKU6SdbYjVhws6Ttp8H','--',
      '--request',body===undefined?'GET':'POST','--header','@'+headerFile,'--dump-header',outputHeaders,
      ...(body===undefined?[]:['--data-binary','@'+bodyFile]),'--silent','--show-error','--write-out','\n%{http_code}'];
    const r=spawnSync(process.execPath,command,{cwd:'apps/sentinel-lab',encoding:'buffer',timeout:120000,maxBuffer:8*1024*1024,
      env:{...process.env,VERCEL_PROJECT_ID:'prj_5aL0COF1DONfPKI4fv25NitAdkOE',VERCEL_ORG_ID:'team_TEsnMvKU6SdbYjVhws6Ttp8H'}});
    if(r.status!==0||!r.stdout)throw Error('Hosted TEST API transport failed; no automatic retry.');
    const split=r.stdout.lastIndexOf(10),status=Number(r.stdout.subarray(split+1).toString()),bytes=r.stdout.subarray(0,split);
    let data:any;try{data=JSON.parse(bytes.toString());}catch{/* ZIP response. */}
    const responseHeaders=await readFile(outputHeaders,'utf8');
    const cookie=responseHeaders.match(/^set-cookie:\s*(__Host-eidos_session=[a-f0-9]{64}); Path=\/; HttpOnly; SameSite=Lax; Max-Age=2592000; Secure\s*$/im)?.[1];
    return {status,data,bytes,cookie};
  }finally{for(const file of [headerFile,bodyFile,outputHeaders])try{await unlink(file);}catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}}
}
const receipt:Record<string,any>={timestampUtc:new Date().toISOString(),phase:args.phase,deploymentId:deployment.deployment.id,backendSha:deployment.backendSha,origin,
  openaiCalls:0,syntheticSessions:false,syntheticPaidRows:false,secretValuesPrinted:false};
let file='hosted-'+args.phase+'-'+Date.now()+'.json';
try {
  if(args.phase==='provider-metadata') {
    const r=await call('/api/playground-provider-check',undefined,undefined,true);receipt.status=r.status;receipt.provider=r.data;receipt.openaiCalls=1;
    if(r.status!==200||!r.data?.authenticated||r.data.inferenceRequests!==0||r.data.countRequests!==0)throw Error('Runtime readiness mismatch.');
  } else if(args.phase==='bindings') {
    const r=await call('/api/playground-provider-check',undefined,undefined,true);receipt.status=r.status;receipt.runtime=r.data;
    if(r.status!==200)throw Error('Runtime readiness mismatch.');
  } else if(args.phase==='preflight') {
    const r=await call('/api/public-config');receipt.config=r.data;receipt.status=r.status;
    if(r.status!==200||r.data.runtimeSourceRevision!==deployment.backendSha||r.data.runtimeBundleHash!==deployment.bundleHash||r.data.accountsReady!==true||r.data.shopReady!==true||r.data.playgroundAiReady!==false)throw Error('Runtime readiness mismatch.');
  } else if(args.phase==='signin') {
    receipt.attempts=[];
    if(args.slot&&!['business','personal'].includes(args.slot))throw Error('Choose an owned inbox.');
    for(const [slot,actor] of Object.entries(actors)) {
      if(args.slot&&args.slot!==slot)continue;
      if(state[slot]?.cookie){receipt.attempts.push({slot,alreadyAuthenticated:true});continue;}
      const r=await call('/api/members/auth',{...actor,kind:'person',newsletter:false,challenge:'XXXX.DUMMY.TOKEN.XXXX'});
      receipt.attempts.push({slot,status:r.status,error:r.data?.error||null,mailAccepted:r.status===200});
      if(r.status!==200)throw Error('Normal sign-in mail was not accepted.');
    }
  } else if(args.phase==='verify') {
    const slot=args.slot;if(!['business','personal'].includes(slot))throw Error('Choose an owned inbox.');
    if(state[slot]?.cookie)throw Error('This owned inbox already has a genuine session.');
    const token=await privateInput();
    const r=await call('/api/members/auth',{action:'verify',token});
    receipt.slot=slot;receipt.status=r.status;receipt.cookiePresent=Boolean(r.cookie);
    if(r.status!==200||!r.cookie||!r.data?.member?.username)throw Error('Normal one-time verification did not establish a session.');
    // Public members omit database IDs. Read the genuine newly issued session only,
    // and check ownership through the normal authenticated account endpoint.
    state[slot]={cookie:r.cookie};await saveState();
    const account=await call('/api/members/account',undefined,slot);receipt.accountStatus=account.status;
    if(account.status!==200||account.data?.member?.email!==actors[slot].email||account.data?.member?.username!==r.data.member.username)throw Error('The genuine session does not match the requested owned inbox.');
    const {client}=await testDatabase();
    try {
      const sessionHash=sha(r.cookie.slice('__Host-eidos_session='.length));
      const result=await client.execute({sql:'SELECT s.member_id,m.email,m.username FROM eidos_member_sessions s JOIN eidos_email_members m ON m.id=s.member_id WHERE s.token_hash=? AND s.expires>? AND m.disabled=0',args:[sessionHash,Math.floor(Date.now()/1000)]});
      const row=result.rows[0];
      if(result.rows.length!==1||row.email!==actors[slot].email||row.username!==r.data.member.username||typeof row.member_id!=='string')throw Error('The issued session owner could not be verified.');
      state[slot].memberId=row.member_id;receipt.memberIdSha256=sha(row.member_id);await saveState();
    } finally {client.close();}
    const replay=await call('/api/members/auth',{action:'verify',token});
    receipt.consumedLinkReplayStatus=replay.status; if(replay.status!==400)throw Error('The one-time link was reusable.');
  } else if(args.phase==='freeze') {
    if(!state.business?.cookie||!state.personal?.cookie||state.business.memberId===state.personal.memberId)throw Error('Two genuinely authenticated owned inboxes are required.');
    const snapshotPath=path.join(root,'approved-public-snapshot.json');let snapshot;
    try{snapshot=JSON.parse(await readFile(snapshotPath,'utf8'));}catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}
    if(!snapshot){const saved=JSON.parse(await readFile(path.resolve('artifacts/playground-authenticated-20261015/generation-v1/generation-ledger-readback.json'),'utf8'));
      const project={...upgradeProject(applyContextProposal(ownerCountRequest().context,saved.generation.proposal)),schemaVersion:5 as const};
      snapshot=await createProjectSnapshot(project,'hosted-acceptance-public-sample',1,'switchboard-developer-v1');
      await writeFile(snapshotPath,JSON.stringify(snapshot,null,2)+'\n',{flag:'wx'});}
    const expected=await buildFrozenPackage(snapshot);
    await writeFile(path.join(root,'compiled-server-approved.zip'),expected.archive);
    state.deliveryRequestId||=crypto.randomUUID();await saveState();
    const r=await call('/api/playground/delivery',{requestId:state.deliveryRequestId,snapshot},'business');
    if(r.status!==200||r.data?.delivery?.archiveSha256!==expected.receipt.archiveSha256)throw Error('Hosted approved package differs from the local renderer.');
    state.delivery=r.data.delivery;await saveState();receipt.delivery=r.data.delivery;
    const again=await call('/api/playground/delivery',{requestId:state.deliveryRequestId,snapshot},'business');
    const other=await call('/api/playground/delivery?id='+state.delivery.id,undefined,'personal');
    const anonymous=await call('/api/playground/delivery?id='+state.delivery.id);
    receipt.idempotent=again.status===200&&again.data?.delivery?.id===state.delivery.id;receipt.otherOwnerStatus=other.status;receipt.anonymousStatus=anonymous.status;
    if(!receipt.idempotent||other.status!==404||anonymous.status!==401)throw Error('Frozen package ownership failed.');
    const caps=[];for(const slot of ['business','personal']){const result=await call('/api/playground/ai',{action:'limit',limitMicros:1000000},slot);caps.push({slot,status:result.status,balance:result.data?.balance});if(result.status!==200||result.data?.balance?.availableIncluded!==0||result.data?.balance?.availablePurchased!==0)throw Error('Account cap or funding boundary failed.');}receipt.caps=caps;
  } else if(args.phase==='checkout') {
    if(!state.delivery||!state.business?.cookie)throw Error('Freeze the approved package from a genuine account first.');
    state.checkoutAttempt||=randomBytes(32).toString('hex');await saveState();
    const r=await call('/api/shop/checkout',{productId:'switchboard-developer-v1',editionId:'developer',acceptTerms:true,attempt:state.checkoutAttempt,deliveryId:state.delivery.id,challenge:'XXXX.DUMMY.TOKEN.XXXX'},'business');
    receipt.status=r.status;receipt.error=r.data?.error||null;
    if(r.status!==200||!r.data?.url||new URL(r.data.url).origin!=='https://checkout.stripe.com')throw Error('Genuine TEST checkout was not prepared.');
    state.checkout=r.data;await saveState();receipt.checkout={orderId:r.data.orderId||null,provider:'Stripe',mode:'test',amountCents:9900,prepared:true,paid:false};
    const pending=await call('/api/shop/status',{receipt:state.checkoutAttempt},'business');
    receipt.pending={status:pending.status,orderStatus:pending.data?.status,downloadTokenPresent:Boolean(pending.data?.downloadToken)};
    if(pending.status!==200||pending.data?.status!=='pending'||pending.data?.downloadToken)throw Error('Unpaid checkout granted entitlement.');
  } else if(args.phase==='payment') {
    if(!state.checkoutAttempt||!state.delivery)throw Error('Prepare the genuine TEST checkout first.');
    const paid=await call('/api/shop/status',{receipt:state.checkoutAttempt},'business');
    receipt.payment={httpStatus:paid.status,status:paid.data?.status,fulfillment:paid.data?.fulfillmentStatus,mail:paid.data?.mailStatus,approvedPage:paid.data?.approvedPage};
    if(paid.status!==200||paid.data?.status!=='paid'||paid.data?.approvedPage?.state!=='ready'||!/^[a-f0-9]{64}$/.test(paid.data?.downloadToken||''))throw Error('Genuine signed payment has not granted the approved package yet.');
    state.orderId=paid.data.orderId;state.downloadToken=paid.data.downloadToken;await saveState();
    const other=await call('/api/shop/status',{receipt:state.checkoutAttempt},'personal');receipt.otherOwnerStatus=other.status;
    const approved=await call('/api/shop/download',{receipt:state.checkoutAttempt,downloadToken:state.downloadToken,artifact:'approved-page'},'business');
    const original=await call('/api/shop/download',{receipt:state.checkoutAttempt,downloadToken:state.downloadToken,artifact:'original'},'business');
    const otherDownload=await call('/api/shop/download',{receipt:state.checkoutAttempt,downloadToken:state.downloadToken,artifact:'approved-page'},'personal');
    const anonymousDownload=await call('/api/shop/download',{receipt:state.checkoutAttempt,downloadToken:state.downloadToken,artifact:'approved-page'});
    const snapshot=JSON.parse(await readFile(path.join(root,'approved-public-snapshot.json'),'utf8'));
    receipt.downloads={approved:{status:approved.status,bytes:approved.bytes.length,sha256:sha(approved.bytes)},original:{status:original.status,bytes:original.bytes.length,sha256:sha(original.bytes)},otherOwnerStatus:otherDownload.status,anonymousStatus:anonymousDownload.status};
    if(other.status!==403||approved.status!==200||sha(approved.bytes)!==state.delivery.archiveSha256||original.status!==200||sha(original.bytes)!==snapshot.parent.sha256||original.bytes.length!==snapshot.parent.bytes||otherDownload.status!==403||![401,403].includes(anonymousDownload.status))throw Error('Paid delivery bytes or ownership failed.');
    await writeFile(path.join(root,'hosted-approved-public-sample.zip'),approved.bytes);
  } else if(args.phase==='fund') {
    if(!state.orderId||!state.business?.memberId)throw Error('Verify the genuine paid account purchase first.');
    const {client,env}=await testDatabase();
    try {
      const before=await aiBalance(env,state.business.memberId,'test');
      if(before.requestedLimit!==1000000)throw Error('The explicitly chosen TEST monthly cap changed.');
      const row=await env.EIDOS_DB.prepare('SELECT o.id,t.product_id,t.edition_id,t.version,t.price_cents FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id WHERE o.id=? AND t.member_id=?').bind(state.orderId,state.business.memberId).first<Record<string,any>>();
      if(!row||row.product_id!=='switchboard-developer-v1'||row.edition_id!=='developer'||row.version!=='1.0.3'||row.price_cents!==9900)throw Error('A matching immutable purchase is required.');
      receipt.grant=await grantPaidAiBalance(env,state.business.memberId,'test',row.id,{productId:row.product_id,editionId:row.edition_id,version:row.version,priceCents:row.price_cents,grantVersion:'playground-owner-test-20261015-v1',kind:'included',amount:1000000});
      receipt.balance=await aiBalance(env,state.business.memberId,'test');receipt.policy=await aiPolicyStatus(env,'test');
      if(receipt.balance.requestedLimit!==before.requestedLimit)throw Error('The grant changed the cap.');
    } finally {client.close();}
  } else if(args.phase==='recover') {
    const r=await call('/api/shop/recover',{email:actors.business.email,challenge:'XXXX.DUMMY.TOKEN.XXXX'},'business');receipt.status=r.status;
    if(r.status!==200)throw Error('Normal purchase recovery was not accepted.');
  } else if(args.phase==='redeem') {
    const token=await privateInput();const r=await call('/api/shop/redeem',{token},'business');
    receipt.status=r.status;receipt.approvedPage=r.data?.approvedPage;
    if(r.status!==200||!/^[a-f0-9]{64}$/.test(r.data?.receipt||'')||r.data?.approvedPage?.deliveryId!==state.delivery.id||r.data?.approvedPage?.archiveSha256!==state.delivery.archiveSha256)throw Error('Inbox recovery did not retain the same approved package.');
    state.checkoutAttempt=r.data.receipt;await saveState();
    const replay=await call('/api/shop/redeem',{token},'business');receipt.consumedRecoveryReplayStatus=replay.status;
    if(replay.status!==401)throw Error('The recovery link was reusable.');
  } else if(args.phase==='refund') {
    if(!state.orderId||!state.business?.memberId)throw Error('Only the newly accepted TEST order may be refunded.');
    const {client,configuration,env}=await testDatabase();
    try {
      if(!/^[sr]k_test_/.test(configuration.STRIPE_SECRET_KEY||''))throw Error('An approved Stripe TEST key is required.');
      const row=await env.EIDOS_DB.prepare("SELECT o.payment_intent,o.session_id FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id WHERE o.id=? AND t.member_id=? AND o.status='paid'").bind(state.orderId,state.business.memberId).first<{payment_intent:string;session_id:string}>();
      if(!row?.session_id.startsWith('cs_test_')||!row.payment_intent.startsWith('pi_'))throw Error('A genuine TEST paid order is required.');
      const headers={authorization:'Bearer '+configuration.STRIPE_SECRET_KEY,'stripe-version':'2025-06-30.basil'};
      const accountResponse=await fetch('https://api.stripe.com/v1/account',{headers,redirect:'error'}),account=await accountResponse.json();
      if(!accountResponse.ok||account.id!=='acct_1Rj03gKC8pRG5Tr9')throw Error('The approved TEST merchant changed.');
      const r=await fetch('https://api.stripe.com/v1/refunds',{method:'POST',redirect:'error',signal:AbortSignal.timeout(15000),headers:{...headers,'content-type':'application/x-www-form-urlencoded','idempotency-key':'eidos-playground-acceptance-refund-'+state.orderId},body:new URLSearchParams({payment_intent:row.payment_intent})});
      const value=await r.json();if(!r.ok||!value.id?.startsWith('re_')||value.status!=='succeeded'||value.amount!==9900||value.payment_intent!==row.payment_intent)throw Error('The genuine TEST refund did not succeed.');
      receipt.refund={id:value.id,status:value.status,amountCents:value.amount,currency:value.currency,testSession:true};
    } finally {client.close();}
  } else if(args.phase==='revocation') {
    const status=await call('/api/shop/status',{receipt:state.checkoutAttempt},'business');receipt.order={status:status.status,paymentStatus:status.data?.status,approvedPage:status.data?.approvedPage,downloadTokenPresent:Boolean(status.data?.downloadToken)};
    const downloads=[];for(const artifact of ['approved-page','original']){const r=await call('/api/shop/download',{receipt:state.checkoutAttempt,downloadToken:state.downloadToken,artifact},'business');downloads.push({artifact,status:r.status});}receipt.downloads=downloads;
    const balance=await call('/api/playground/ai',undefined,'business');receipt.balance=balance.data?.balance;
    if(status.status!==200||status.data?.status!=='refunded'||status.data?.approvedPage?.state!=='revoked'||status.data?.downloadToken||downloads.some(r=>r.status!==403)||!receipt.balance?.frozen)throw Error('Signed refund revocation or AI funding reversal is incomplete.');
  } else if(args.phase==='ledger') {
    const {client,env}=await testDatabase();try{
      receipt.policy=await aiPolicyStatus(env,'test');receipt.uninitializedTables=[];
      receipt.counts=Object.fromEntries(await Promise.all(['eidos_email_members','eidos_orders','eidos_template_orders','eidos_pg_delivery_snapshots','eidos_pg_delivery_orders','eidos_pg_ai_actions','eidos_pg_ai_grants'].map(async name=>{
        const exists=await client.execute({sql:"SELECT name FROM sqlite_master WHERE type='table' AND name=?",args:[name]});
        if(!exists.rows.length){receipt.uninitializedTables.push({table:name,reason:'Created by the first genuine authorized delivery request; no rows seeded for acceptance.'});return [name,null];}
        return [name,Number((await client.execute('SELECT COUNT(*) n FROM '+name)).rows[0].n)];
      })));
    }finally{client.close();}
  } else throw Error('Choose a supported acceptance phase.');
  receipt.passed=true;
}catch(error){receipt.passed=false;receipt.failure=error instanceof Error&&/^(Runtime readiness mismatch|Normal sign-in mail|Hosted TEST API transport|Hosted approved package|Frozen package ownership|Account cap or funding|Genuine TEST checkout|Unpaid checkout|Unexpected API route)/.test(error.message)?error.message:'Hosted acceptance did not pass. Private tokens and response details suppressed.';process.exitCode=1;}
await writeFile(path.join(root,file),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
