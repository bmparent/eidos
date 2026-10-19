/** Toggle only this task's TEST endpoint; activation requires the published paired relay. */
import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { testSite } from './deploy_playground_test_candidate.mjs';
export function validateTestWebhook(endpoint,stored) {
  if(endpoint?.id!==stored?.endpointId||endpoint.livemode!==false||endpoint.url!==testSite+'/api/shop/webhook'||!['enabled','disabled'].includes(endpoint.status))throw Error('Only the exact owned TEST endpoint may change.');
}
async function main() {
  const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
  if(!args.configuration||!['enabled','disabled'].includes(args.state))throw Error('Supply the approved ignored TEST configuration and --state=enabled|disabled.');
  const {env}=JSON.parse(await readFile(args.configuration,'utf8'));
  if(env.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!/^[sr]k_test_/.test(env.STRIPE_SECRET_KEY||''))throw Error('Only the approved TEST configuration is allowed.');
  const stored=JSON.parse(await readFile('apps/sentinel-lab/.vercel/playground-test.private.json','utf8'));
  if(!/^we_[A-Za-z0-9]+$/.test(stored.endpointId)||stored.siteOrigin!==testSite)throw Error('Unexpected stored TEST endpoint.');
  const stripe=async(route,body)=>{
    const r=await fetch('https://api.stripe.com/v1/'+route,{method:body?'POST':'GET',redirect:'error',signal:AbortSignal.timeout(15000),headers:{authorization:'Bearer '+env.STRIPE_SECRET_KEY,'stripe-version':'2025-06-30.basil',...(body?{'content-type':'application/x-www-form-urlencoded','idempotency-key':randomUUID()}:{})},...(body?{body:new URLSearchParams(body)}:{})});
    const result=await r.json();if(!r.ok)throw Error('Stripe TEST endpoint request rejected ('+r.status+').');return result;
  };
  const account=await stripe('account');if(account.id!=='acct_1Rj03gKC8pRG5Tr9')throw Error('Unexpected TEST merchant.');
  const before=await stripe('webhook_endpoints/'+stored.endpointId);validateTestWebhook(before,stored);
  if(args.state==='enabled') {
    const pin=JSON.parse(await readFile('artifacts/playground-authenticated-20261015/release-continuation/test-backend-deployment.json','utf8'));
    const r=await fetch(testSite+'/api/public-config',{redirect:'error',cache:'no-store',signal:AbortSignal.timeout(20000)});
    const config=await r.json();
    if(!r.ok||config.accountsReady!==true||config.shopReady!==true||config.playgroundAiReady!==false||config.runtimeSourceRevision!==pin.backendSha||config.runtimeBundleHash!==pin.bundleHash)throw Error('Publish and verify the exact account-ready TEST relay before enabling delivery.');
    const hook=await fetch(testSite+'/api/shop/webhook',{method:'POST',redirect:'error',body:'{}',headers:{'content-type':'application/json'}});
    if(hook.status!==400)throw Error('The published TEST signature boundary is unavailable.');
  }
  const after=before.status===args.state?before:await stripe('webhook_endpoints/'+stored.endpointId,{disabled:String(args.state==='disabled')});
  validateTestWebhook(after,stored);if(after.status!==args.state)throw Error('TEST endpoint state did not change.');
  const receipt={timestampUtc:new Date().toISOString(),endpointId:stored.endpointId,url:after.url,livemode:false,before:before.status,after:after.status,otherEndpointsModified:false,providerCalls:0};
  await writeFile('artifacts/playground-authenticated-20261015/release-continuation/stripe-test-endpoint-state-'+Date.now()+'.json',JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify(receipt));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(()=>{console.error('Scoped TEST endpoint operation failed; private details suppressed.');process.exitCode=1;});
