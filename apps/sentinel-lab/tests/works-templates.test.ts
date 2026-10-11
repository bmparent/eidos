import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash, createHmac } from 'node:crypto';
import { createClient } from '@libsql/client';
import { adaptDatabase } from '../lib/works/database';
import { challenge, hash, type PlatformEnv } from '../lib/works/vendor/functions/_shared/platform/core';
import { catalog, recover, redeem, templateTestReady } from '../lib/works/vendor/functions/_shared/platform/templateShop';
import { templateProducts } from '../lib/works/vendor/functions/_shared/platform/templateCatalog';
import { onRequestPost as checkout } from '../lib/works/vendor/functions/api/shop/checkout';
import { onRequestPost as status } from '../lib/works/vendor/functions/api/shop/status';
import { onRequestPost as download } from '../lib/works/vendor/functions/api/shop/download';
import { onRequestPost as webhook } from '../lib/works/vendor/functions/api/shop/webhook';
import { kitArchiveBase64 } from '../lib/works/vendor/functions/_shared/platform/kitArchive';

const origin = 'http://localhost:4319';
function request(path: string, payload: unknown, cookie = '') {
  return new Request(origin + path, { method: 'POST', headers: { origin, 'content-type': 'application/json', cookie }, body: JSON.stringify(payload) });
}
async function fixture() {
  const client = createClient({ url: ':memory:' }), database = adaptDatabase(client);
  for (const name of ['0001_eidos_platform.sql', '0002_members.sql', '0004_paid_templates.sql']) {
    const schema = readFileSync(new URL('../lib/works/vendor/migrations/' + name, import.meta.url), 'utf8');
    await client.executeMultiple(schema);
    if (name === '0004_paid_templates.sql') await client.executeMultiple(schema);
  }
  let bytes: Uint8Array = new Uint8Array([80, 75, 3, 4, 1, 2, 3]);
  const env: PlatformEnv = { EIDOS_DB: database, EIDOS_SHOP_ENABLED: 'true', EIDOS_LOCAL_TEST: 'true',
    EIDOS_TEMPLATE_TEST_ENABLED: 'true', EIDOS_TEMPLATE_VERIFIED_EDITIONS: 'switchboard-developer-v1,switchboard-wordpress-v1',
    STRIPE_SECRET_KEY: 'sk_test_local_fixture', EIDOS_KIT_WEBHOOK_SECRET: 'whsec_local_fixture_only', PUBLIC_SITE_URL: origin,
    EIDOS_ACCOUNTS_ENABLED: 'true', RESEND_API_KEY: 'fixture-mail-key', EIDOS_MAIL_FROM: 'Eidos <hello@example.test>',
    EIDOS_TEMPLATE_ARCHIVES: { async get(key) { return key.startsWith('templates/') ? { bytes, sha256: createHash('sha256').update(bytes).digest('hex') } : null; } } };
  const originalFetch = globalThis.fetch, mails: Array<{text:string;to:string;idempotency:string}> = [], sessions: URLSearchParams[] = [];
  let failMail = false, failCheckout = false;
  globalThis.fetch = async (url, init) => {
    if (String(url) === 'https://api.stripe.com/v1/checkout/sessions') {
      if (failCheckout) return Response.json({error:'fixture'}, {status:503});
      const body = new URLSearchParams(String(init?.body)), key = body.get('client_reference_id')!;
      assert.equal(body.has('payment_method_types[0]'),false);
      assert.equal(new Headers(init?.headers).get('stripe-version'),'2025-06-30.basil');
      sessions.push(body);
      return Response.json({ id: 'cs_test_' + key.replaceAll('-', ''), url: 'https://checkout.stripe.com/c/pay/cs_test_' + key, livemode: false });
    }
    if (String(url) === 'https://api.resend.com/emails') {
      const mail = JSON.parse(String(init?.body));
      mails.push({ ...mail, idempotency: new Headers(init?.headers).get('idempotency-key')! });
      return Response.json(failMail ? {} : { id: 'mail_fixture_' + mails.length }, {status:failMail ? 503 : 200});
    }
    throw Error('Unexpected network in controlled test: ' + url);
  };
  async function call(handler: (value: { request: Request; env: PlatformEnv }) => Promise<Response>, path: string, payload: unknown, cookie = '') {
    return handler({ request: request(path, payload, cookie), env });
  }
  async function begin(attempt: string, editionId = 'developer', cookie = '', slug = 'switchboard') {
    const response = await call(checkout, '/api/shop/checkout', { productId: slug + '-' + editionId + '-v1', editionId, acceptTerms:true, attempt }, cookie);
    assert.equal(response.status, 200, await response.clone().text());
    const row = await database.prepare('SELECT o.*,t.* FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id WHERE o.receipt_hash=?').bind(await hash(attempt)).first<Record<string,unknown>>();
    assert.ok(row); return row;
  }
  async function event(row: Record<string,unknown>, id: string, overrides: Record<string,unknown> = {}, type = 'checkout.session.completed', livemode = false, signature = true) {
    const payload = {id, type, livemode, data:{object:{ id:row.session_id, client_reference_id:row.id, mode:'payment', currency:'usd', amount_total:row.price_cents,
      payment_status:'paid', payment_intent:'pi_test_' + row.id, customer_details:{email:'buyer@example.test'},
      metadata:{eidos_product:row.product_id,eidos_edition:row.edition_id,eidos_version:row.version,eidos_order_id:row.id}, ...overrides }}};
    const raw = JSON.stringify(payload), timestamp = Math.floor(Date.now()/1000);
    const digest = createHmac('sha256', env.EIDOS_KIT_WEBHOOK_SECRET!).update(timestamp + '.' + raw).digest('hex');
    return webhook({ env, request: new Request(origin + '/api/shop/webhook', {method:'POST', headers:{'stripe-signature':`t=${timestamp},v1=${signature ? digest : '0'.repeat(64)}`},body:raw}) });
  }
  return { env, database, client, call, begin, event, mails, sessions, setBytes(value:Uint8Array){bytes=value;}, setFailMail(value:boolean){failMail=value;}, setFailCheckout(value:boolean){failCheckout=value;}, close(){globalThis.fetch=originalFetch;client.close();} };
}

test('template TEST gate rejects LIVE, production host, missing archives and unverified CMS; server prices cannot be overridden', async () => {
  const f=await fixture();
  try {
    assert.equal(templateProducts.length,8);
    assert.equal(templateTestReady({...f.env,STRIPE_SECRET_KEY:'sk_live_forbidden'}),false);
    assert.equal(templateTestReady({...f.env,PUBLIC_SITE_URL:'https://eidos-works.com'}),false);
    for (const host of ['www.eidos-works.com','owner-preview.eidos-works.com']) assert.equal(templateTestReady({...f.env,PUBLIC_SITE_URL:'https://'+host}),false);
    assert.equal(templateTestReady({...f.env,EIDOS_TEMPLATE_TEST_ENABLED:'false'}),false);
    for (const payload of [
      {productId:'switchboard-developer-v1',editionId:'developer',priceCents:1},
      {productId:'switchboard-developer-v1',editionId:'wordpress'},
      {productId:'switchboard-squarespace-v1',editionId:'squarespace'},
    ]) assert.equal((await f.call(checkout,'/api/shop/checkout',{...payload,acceptTerms:true})).status,400);
    f.env.EIDOS_TEMPLATE_VERIFIED_EDITIONS='';
    assert.equal((await f.call(checkout,'/api/shop/checkout',{productId:'switchboard-wordpress-v1',editionId:'wordpress',acceptTerms:true})).status,503);
    const catalogResult=await (await catalog({env:f.env,request:new Request(origin+'/api/shop/catalog')})).json();
    assert.equal(catalogResult.editions.every((item:{available:boolean})=>!item.available),true);
    assert.equal(f.sessions.length,0);
  } finally {f.close();}
});

test('catalog updates all eight edition versions while all sixteen prior purchase snapshots retain their original archive', async () => {
  const f=await fixture();
  try {
    for(const product of templateProducts) {
      const expected=product.editionId==='wordpress'||product.id==='switchboard-developer-v1'?'1.0.2':product.id==='nightjar-developer-v1'?'1.0.4':'1.0.1';
      assert.equal(product.version,expected);
      assert.match(product.archiveKey,new RegExp('/'+product.version.replaceAll('.','\\.')+'/'+product.editionId+'\\.zip$'));
      assert.ok(product.downloadName.endsWith('-'+product.version+'.zip'));
    }
    f.env.EIDOS_TEMPLATE_VERIFIED_EDITIONS=templateProducts.map(product=>product.id).join(',');
    const retained: Array<[string,string,string]> = [
      ...['switchboard','tideglass','matter','nightjar'].flatMap(slug=>[['developer','1.0.0'],['wordpress','1.0.0'],['wordpress','1.0.1']].map(([edition,version])=>[slug,edition,version] as [string,string,string])),
      ['switchboard','developer','1.0.1'],
      ...['1.0.1','1.0.2','1.0.3'].map(version=>['nightjar','developer',version] as [string,string,string]),
    ];
    assert.equal(retained.length,16);
    for(const [index,[slug,edition,oldVersion]] of retained.entries()) {
      const receipt=index.toString(16).repeat(64),current=await f.begin(receipt,edition,'',slug);
      assert.equal(current.version,templateProducts.find(product=>product.id===`${slug}-${edition}-v1`)!.version);
      // Simulate a retained order created by the previous catalog revision.
      const oldBytes=new Uint8Array([80,75,3,4,9,8,7]),oldHash=createHash('sha256').update(oldBytes).digest('hex');
      const original={...current,version:oldVersion,archive_key:`templates/${slug}/${oldVersion}/${edition}.zip`,archive_sha256:oldHash,download_name:`eidos-${slug}-${edition}-${oldVersion}.zip`};
      await f.database.prepare('UPDATE eidos_template_orders SET version=?,archive_key=?,archive_sha256=?,download_name=? WHERE order_id=?').bind(original.version,original.archive_key,original.archive_sha256,original.download_name,current.id).run();
      assert.equal((await f.event(original,'evt_retained_original_version_'+slug+'_'+edition+'_'+oldVersion.replaceAll('.','_'))).status,200);
      const paid=await (await f.call(status,'/api/shop/status',{receipt})).json();
      assert.equal(paid.version,oldVersion);assert.equal(paid.downloadName,original.download_name);
      const keys:string[]=[];const originalArchive=f.env.EIDOS_TEMPLATE_ARCHIVES!;
      f.env.EIDOS_TEMPLATE_ARCHIVES={async get(key){keys.push(key);return key===original.archive_key?{bytes:oldBytes,sha256:oldHash}:originalArchive.get(key);}};
      const zip=await f.call(download,'/api/shop/download',{receipt,downloadToken:paid.downloadToken});
      assert.equal(zip.status,200);assert.deepEqual(keys,[original.archive_key]);
      assert.equal(zip.headers.get('x-eidos-archive-sha256'),oldHash);
      assert.deepEqual(new Uint8Array(await zip.arrayBuffer()),oldBytes);
      assert.ok(zip.headers.get('content-disposition')!.includes(original.download_name));
      f.env.EIDOS_TEMPLATE_ARCHIVES={async get(){return null;}};
      assert.equal((await f.call(download,'/api/shop/download',{receipt,downloadToken:paid.downloadToken})).status,503);
      f.env.EIDOS_TEMPLATE_ARCHIVES=originalArchive;
    }
  } finally {f.close();}
});

test('official public TEST challenge still calls siteverify and is confined to exact candidate commerce actions', async () => {
  const f=await fixture();let calls=0;
  try {
    const publicOrigin='https://eidosworks-templates-test-20261010.pages.dev';
    const env={...f.env,PUBLIC_SITE_URL:publicOrigin,EIDOS_TEMPLATE_TEST_CHALLENGE:'true',TURNSTILE_SITE_KEY:'1x00000000000000000000AA',TURNSTILE_SECRET_KEY:'1x0000000000000000000000000000000AA'};
    const req=new Request(publicOrigin+'/api/shop/checkout');
    globalThis.fetch=async()=>{calls++;return Response.json({success:true,hostname:'example.com',metadata:{result_with_testing_key:true}});};
    await challenge(req,env,'XXXX.DUMMY.TOKEN.XXXX','checkout');assert.equal(calls,1);
    for(const [request,configuration,action] of [
      [req,{...env,EIDOS_TEMPLATE_TEST_CHALLENGE:'false'},'checkout'],
      [req,{...env,STRIPE_SECRET_KEY:'sk_live_forbidden'},'checkout'],
      [new Request('https://www.eidos-works.com/api/shop/checkout'),env,'checkout'],
      [req,env,'newsletter'],
    ] as const) await assert.rejects(challenge(request,configuration,'XXXX.DUMMY.TOKEN.XXXX',action));
    globalThis.fetch=async()=>Response.json({success:false,hostname:'example.com',metadata:{result_with_testing_key:true}});
    await assert.rejects(challenge(req,env,'XXXX.DUMMY.TOKEN.XXXX','checkout'));
  } finally {f.close();}
});

test('checkout attempt retry preserves one ledger order and same server-owned TEST price', async () => {
  const f=await fixture(), receipt='a'.repeat(64);
  try {
    f.setFailCheckout(true);
    const payload={productId:'switchboard-developer-v1',editionId:'developer',acceptTerms:true,attempt:receipt};
    assert.equal((await f.call(checkout,'/api/shop/checkout',payload)).status,502);
    f.setFailCheckout(false);
    const row=await f.begin(receipt); await f.begin(receipt);
    assert.equal(f.sessions.length,1);
    assert.equal(f.sessions[0].get('line_items[0][price_data][unit_amount]'),'9900');
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_orders').first<{n:number}>())?.n,1);
    assert.equal(row.version,'1.0.2');
    assert.equal((await f.call(download,'/api/shop/download',{receipt})).status,403);
    const pending=await (await f.call(status,'/api/shop/status',{receipt})).json();
    assert.equal(pending.status,'pending'); assert.equal(pending.downloadToken,null);
    assert.equal((await f.call(checkout,'/api/shop/checkout',{...payload,editionId:'wordpress',productId:'switchboard-wordpress-v1'})).status,409);
  } finally {f.close();}
});

test('signed paid fulfillment, delayed state, duplicate webhook, snapshot integrity and expiring download links', async () => {
  const f=await fixture(), receipt='b'.repeat(64);
  try {
    const row=await f.begin(receipt);
    assert.equal((await f.event(row,'evt_bad_signature',{},undefined,false,false)).status,400);
    assert.equal((await f.event(row,'evt_live_mode',{},undefined,true)).status,400);
    assert.equal((await f.event(row,'evt_wrong_price',{amount_total:1})).status,400);
    assert.equal((await f.event(row,'evt_delayed',{payment_status:'unpaid'})).status,200);
    assert.equal((await (await f.call(status,'/api/shop/status',{receipt})).json()).status,'pending');
    assert.equal((await f.event(row,'evt_paid',{},'checkout.session.async_payment_succeeded')).status,200);
    assert.equal((await f.event(row,'evt_paid')).status,200); assert.equal(f.mails.length,1);
    assert.match(f.mails[0].text,/transactional delivery/);
    const ready=await (await f.call(status,'/api/shop/status',{receipt})).json();
    assert.equal(ready.status,'paid'); assert.equal(ready.mailStatus,'sent'); assert.equal(ready.version,'1.0.2');
    assert.equal((await f.call(download,'/api/shop/download',{receipt})).status,401);
    const zip=await f.call(download,'/api/shop/download',{receipt,downloadToken:ready.downloadToken});
    assert.equal(zip.status,200); assert.equal(zip.headers.get('cache-control'),'private, no-store');
    assert.equal(zip.headers.get('x-eidos-archive-sha256'),row.archive_sha256);
    await f.database.prepare('UPDATE eidos_template_downloads SET expires=0').run();
    assert.equal((await f.call(download,'/api/shop/download',{receipt,downloadToken:ready.downloadToken})).status,401);
    const fresh=await (await f.call(status,'/api/shop/status',{receipt})).json();
    f.setBytes(new Uint8Array([80,75,0]));
    assert.equal((await f.call(download,'/api/shop/download',{receipt,downloadToken:fresh.downloadToken})).status,503);
  } finally {f.close();}
});

test('failed delayed payment and expired checkout are terminal without entitlement or mail', async () => {
  const f=await fixture(),receipt='6'.repeat(64),expiredReceipt='7'.repeat(64);
  try {
    const row=await f.begin(receipt);
    assert.equal((await f.event(row,'evt_failed', {payment_status:'unpaid'}, 'checkout.session.async_payment_failed')).status,200);
    assert.equal((await f.event(row,'evt_failed', {payment_status:'unpaid'}, 'checkout.session.async_payment_failed')).status,200);
    const failed=await(await f.call(status,'/api/shop/status',{receipt})).json();
    assert.equal(failed.paymentState,'failed');assert.equal(failed.downloadToken,null);
    assert.equal((await f.call(download,'/api/shop/download',{receipt})).status,403);
    assert.equal((await f.event(row,'evt_failed_late_paid',{},'checkout.session.async_payment_succeeded')).status,200);
    assert.equal((await(await f.call(status,'/api/shop/status',{receipt})).json()).paymentState,'failed');
    const expired=await f.begin(expiredReceipt);
    assert.equal((await f.event(expired,'evt_expired',{},'checkout.session.expired')).status,200);
    assert.equal((await(await f.call(status,'/api/shop/status',{receipt:expiredReceipt})).json()).paymentState,'expired');
    assert.equal(f.mails.length,0);
  } finally {f.close();}
});

test('receipt email retries keep provider idempotency body stable and paid access survives mail failure',async()=>{
  const f=await fixture(),receipt='c'.repeat(64);
  try{
    const row=await f.begin(receipt);f.setFailMail(true);
    assert.equal((await f.event(row,'evt_mail_retry')).status,503);
    assert.equal((await(await f.call(status,'/api/shop/status',{receipt})).json()).status,'paid');
    f.setFailMail(false);
    assert.equal((await f.event(row,'evt_mail_retry')).status,200);
    assert.equal(f.mails.length,2);assert.equal(f.mails[0].text,f.mails[1].text);assert.equal(f.mails[0].idempotency,f.mails[1].idempotency);
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_stripe_events WHERE id=?').bind('evt_mail_retry').first<{n:number}>())?.n,1);
  }finally{f.close();}
});

test('recovery is enumeration safe, single use and receipt expiry is recoverable; cross-order token fails', async () => {
  const f=await fixture(), receipt='d'.repeat(64), otherReceipt='e'.repeat(64);
  try {
    const row=await f.begin(receipt); await f.event(row,'evt_recovery_paid');
    const other=await f.begin(otherReceipt);await f.event(other,'evt_other_paid',{customer_details:{email:'other@example.test'}});
    const ready=await(await f.call(status,'/api/shop/status',{receipt})).json();
    assert.equal((await f.call(download,'/api/shop/download',{receipt:otherReceipt,downloadToken:ready.downloadToken})).status,401);
    await f.database.prepare('UPDATE eidos_template_orders SET receipt_expires=0 WHERE order_id=?').bind(row.id).run();
    assert.equal((await f.call(status,'/api/shop/status',{receipt})).status,401);
    const known=await f.call(recover,'/api/shop/recover',{email:'buyer@example.test'});
    const unknown=await f.call(recover,'/api/shop/recover',{email:'unknown@example.test'});
    assert.equal(known.status,unknown.status);assert.deepEqual(await known.json(),await unknown.json());
    const token=f.mails.at(-1)!.text.match(/#token=([a-f0-9]{64})/)![1];
    const recovered=await f.call(redeem,'/api/shop/redeem',{token});assert.equal(recovered.status,200,await recovered.clone().text());
    const newReceipt=(await recovered.json()).receipt;
    assert.equal((await f.call(status,'/api/shop/status',{receipt:newReceipt})).status,200);
    assert.equal((await f.call(status,'/api/shop/status',{receipt})).status,404);
    assert.equal((await f.call(redeem,'/api/shop/redeem',{token})).status,401);
    assert.equal((await f.call(status,'/api/shop/status',{receipt:'f'.repeat(64)})).status,404);
  } finally {f.close();}
});

test('two signed-in buyers remain isolated and refund/dispute revokes future access without deleting receipts', async () => {
  const f=await fixture(),receipt='f'.repeat(64), tokenA='1'.repeat(64), tokenB='2'.repeat(64);
  try {
    for(const [id,email,token] of [['a','a@example.test',tokenA],['b','b@example.test',tokenB]]) {
      await f.database.prepare('INSERT INTO eidos_email_members(id,email,username,kind,created_at,newsletter_after) VALUES(?,?,?,?,?,?)').bind(id,email,'user_'+id,'person','2026-10-10','2026-10-10').run();
      await f.database.prepare('INSERT INTO eidos_member_sessions(token_hash,member_id,expires) VALUES(?,?,?)').bind(await hash(token),id,Math.floor(Date.now()/1000)+3600).run();
    }
    const row=await f.begin(receipt,'developer','eidos_session_local='+tokenA);
    assert.equal((await f.event(row,'evt_account_paid',{customer_details:{email:'a@example.test'}})).status,200);
    assert.equal((await f.call(status,'/api/shop/status',{receipt},'eidos_session_local='+tokenB)).status,403);
    const ready=await(await f.call(status,'/api/shop/status',{receipt},'eidos_session_local='+tokenA)).json();
    assert.equal((await f.call(download,'/api/shop/download',{receipt,downloadToken:ready.downloadToken},'eidos_session_local='+tokenB)).status,403);
    const refund=await f.event(row,'evt_refund',{payment_intent:'pi_test_'+row.id},'charge.refunded');assert.equal(refund.status,200);
    assert.equal((await f.call(download,'/api/shop/download',{receipt,downloadToken:ready.downloadToken})).status,403);
    assert.equal((await f.event(row,'evt_late_paid',{customer_details:{email:'a@example.test'}})).status,200);
    const revoked=await(await f.call(status,'/api/shop/status',{receipt})).json();assert.equal(revoked.status,'refunded');assert.equal(revoked.fulfillmentStatus,'revoked');
    assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_template_orders').first<{n:number}>())?.n,1);
  } finally {f.close();}
});

test('signed TEST revocations reconcile template side tables after a legacy hook consumes the shared event ID', async()=>{
  const f=await fixture();
  try {
    for(const [type,letter,otherLetter] of [['charge.refunded','3','4'],['charge.dispute.created','5','0']]) {
      const receipt=letter.repeat(64),otherReceipt=otherLetter.repeat(64);
      const row=await f.begin(receipt),other=await f.begin(otherReceipt);
      assert.equal((await f.event(row,'evt_race_paid_'+letter)).status,200);
      assert.equal((await f.event(other,'evt_race_other_'+letter)).status,200);
      const ready=await(await f.call(status,'/api/shop/status',{receipt})).json();
      assert.ok(ready.downloadToken);assert.ok(ready.downloadExpires*1000>Date.now());
      const beforeRevocation=await f.database.prepare('SELECT o.*,t.* FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id WHERE o.id=?').bind(row.id).first<Record<string,unknown>>();
      const intent='pi_test_'+row.id,eventId='evt_race_revoke_'+letter;
      const object={id:type==='charge.refunded'?'ch_fixture_'+letter:'dp_fixture_'+letter,payment_intent:intent};
      assert.equal((await f.event(row,'evt_race_live_'+letter,object,type,true)).status,400);
      // Even metadata pointing to this order cannot redirect an unrelated intent.
      assert.equal((await f.event(row,'evt_race_unknown_'+letter,{...object,payment_intent:'pi_test_unknown_'+letter},type)).status,200);
      assert.equal((await(await f.call(status,'/api/shop/status',{receipt})).json()).status,'paid');
      // Reproduce the older handler's committed financial revocation + global
      // dedupe entry, without its knowing about the new fulfillment side table.
      await f.database.batch([
        f.database.prepare('INSERT INTO eidos_revoked_payments(payment_intent,received_at) VALUES(?,?)').bind(intent,'2026-10-10'),
        f.database.prepare("UPDATE eidos_orders SET status='refunded' WHERE payment_intent=?").bind(intent),
        f.database.prepare('INSERT INTO eidos_stripe_events(id,received_at) VALUES(?,?)').bind(eventId,'2026-10-10'),
      ]);
      assert.equal((await f.database.prepare('SELECT fulfillment_status FROM eidos_template_orders WHERE order_id=?').bind(row.id).first<{fulfillment_status:string}>())?.fulfillment_status,'ready');
      for(let replay=0;replay<2;replay++) assert.equal((await f.event(row,eventId,object,type)).status,200);
      const revoked=await(await f.call(status,'/api/shop/status',{receipt})).json();
      assert.equal(revoked.status,'refunded');assert.equal(revoked.fulfillmentStatus,'revoked');assert.equal(revoked.downloadToken,null);
      assert.equal((await f.call(download,'/api/shop/download',{receipt,downloadToken:ready.downloadToken})).status,403);
      const retained=await f.database.prepare('SELECT o.*,t.* FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id WHERE o.id=?').bind(row.id).first<Record<string,unknown>>();
      for(const key of ['id','paid_at','payment_intent','product_id','edition_id','version','archive_key','archive_sha256','download_name']) assert.equal(retained?.[key],beforeRevocation?.[key]);
      assert.equal((await f.event(row,'evt_race_late_paid_'+letter)).status,200);
      assert.equal((await(await f.call(status,'/api/shop/status',{receipt})).json()).status,'refunded');
      assert.equal((await(await f.call(status,'/api/shop/status',{receipt:otherReceipt})).json()).status,'paid');
      assert.equal((await f.database.prepare('SELECT COUNT(*) n FROM eidos_stripe_events WHERE id=?').bind(eventId).first<{n:number}>())?.n,1);
    }
  }finally{f.close();}
});

test('existing Cinematic Starter paid receipt still downloads the exact original archive', async()=>{
  const f=await fixture(),receipt='9'.repeat(64);
  try {
    await f.database.prepare("INSERT INTO eidos_orders(id,receipt_hash,status,created_at) VALUES(?,?,'paid',?)").bind('legacy-cinematic',await hash(receipt),'2026-09-01').run();
    const result=await f.call(download,'/api/shop/download',{receipt});assert.equal(result.status,200);
    assert.deepEqual(Buffer.from(await result.arrayBuffer()),Buffer.from(kitArchiveBase64,'base64'));
    assert.equal((await(await f.call(status,'/api/shop/status',{receipt})).json()).orderId,'legacy-cinematic');
  }finally{f.close();}
});
