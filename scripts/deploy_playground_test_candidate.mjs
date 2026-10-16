/** Isolated TEST candidate. No branch-global environment writes or production aliases. */
import { readFile, writeFile, mkdir, lstat } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { vercelApi } from '../apps/sentinel-lab/scripts/vercel-preview-client.mjs';
import { verifyPrivateTemplateArchives } from '../apps/sentinel-lab/scripts/check-template-archives.mjs';

export const testSite = 'https://eidosworks-playground-test-20261015.pages.dev';
const branch = 'codex/playground-unified-20261014', project = 'prj_5aL0COF1DONfPKI4fv25NitAdkOE', team = 'team_TEsnMvKU6SdbYjVhws6Ttp8H';
const merchant = 'acct_1Rj03gKC8pRG5Tr9', app = 'apps/sentinel-lab/', output = 'artifacts/playground-authenticated-20261015/release-continuation';
const privatePath = app + '.vercel/playground-test.private.json';
const digest = data => createHash('sha256').update(data).digest('hex');
export function testEnvironment(configuration, relay) {
  if (configuration.EIDOS_DATABASE_BINDING_PREFIX !== 'EIDOS_TEMPLATES_TEST' ||
      !new URL(configuration.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation') ||
      !/^[sr]k_test_/.test(configuration.STRIPE_SECRET_KEY || '') || (relay?.length || 0) < 32)
    throw Error('Only the approved TEST database, merchant key and existing relay are allowed.');
  const env = { ...configuration, PUBLIC_SITE_URL: testSite, EIDOS_PREVIEW_ORIGINS: testSite, EIDOS_PLATFORM_TOKEN: relay,
    EIDOS_RUNTIME: 'sentinel', EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS: 'test-auth', EIDOS_ACCOUNTS_ENABLED: 'true', EIDOS_SHOP_ENABLED: 'true', EIDOS_TEMPLATE_TEST_ENABLED: 'true',
    EIDOS_TEMPLATE_LIVE_ENABLED: 'false', EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID: merchant, EIDOS_TEMPLATE_TEST_CHALLENGE: 'true',
    TURNSTILE_SITE_KEY: '1x00000000000000000000AA', TURNSTILE_SECRET_KEY: '1x0000000000000000000000000000000AA',
    EIDOS_PLAYGROUND_DELIVERY_MODE: 'test', EIDOS_PLAYGROUND_AI_MODE: 'test', EIDOS_PLAYGROUND_AI_ENABLED: 'false',
    EIDOS_PLAYGROUND_INPUT_COUNT_PRICING: '', EIDOS_AI_ENABLED: 'false', OPENAI_API_KEY: '', EIDOS_ASSISTANT_MODEL: '',
    EIDOS_PLAYGROUND_STRIPE_KEY: '', EIDOS_PLAYGROUND_WEBHOOK_SECRET: '', EIDOS_PLAYGROUND_TEST_PRICE_CENTS: '',
    EIDOS_COMMUNITY_STUDIO_ENABLED: 'false', EIDOS_PROACTIVE_ENABLED: 'false', EIDOS_NEWSLETTER_ENABLED: 'false',
    EIDOS_ADMIN_TOKEN: '', EIDOS_MAINTENANCE_TOKEN: '', EIDOS_MAIL_DAILY_LIMIT: '30', EIDOS_LOCAL_TEST: 'false' };
  // Inherit only the already authorized Preview branch secrets. Never export/decrypt either key.
  delete env.RESEND_API_KEY; delete env.EIDOS_PLAYGROUND_OPENAI_API_KEY;
  return env;
}
async function main() {
  const args = Object.fromEntries(process.argv.slice(2).filter(s => s.startsWith('--') && s.includes('=')).map(s => { const i=s.indexOf('='); return [s.slice(2,i),s.slice(i+1)]; }));
  const git = (...a) => execFileSync('git', a, { encoding:'utf8' }).trim();
  if (!args.cli || git('branch','--show-current') !== branch) throw Error('Use the authenticated CLI from the isolated Playground branch.');
  const receiptPath = output + '/test-backend-deployment.json';
  if (process.argv.includes('--inspect')) {
    const receipt=JSON.parse(await readFile(receiptPath,'utf8'));
    const r=vercelApi(args.cli,'/v13/deployments/'+receipt.deployment.id+'?teamId='+team);
    if(r.projectId!==project || r.target==='production' || r.meta?.paidTemplatesGitSha!==receipt.backendSha || r.meta?.paidTemplatesSourceSha256!==receipt.bundleHash || r.meta?.playgroundTestCandidate!=='true') throw Error('Unexpected deployment identity.');
    receipt.deployment.state=r.readyState; receipt.inspectedAt=new Date().toISOString();
    await writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n'); console.log(JSON.stringify(receipt)); return;
  }
  if(!args.configuration || git('diff','HEAD','--',app)) throw Error('Supply the approved ignored TEST configuration and commit the reviewed application source.');
  const configuration=JSON.parse(await readFile(args.configuration,'utf8')).env;
  const relay=JSON.parse(await readFile(app+'.vercel/playground-manual.private.json','utf8')).env.EIDOS_PLATFORM_TOKEN;
  const env=testEnvironment(configuration,relay), sha=git('rev-parse','HEAD');
  const bindings=vercelApi(args.cli,'/v10/projects/'+project+'/env?decrypt=false&teamId='+team);
  for(const name of ['RESEND_API_KEY','EIDOS_PLAYGROUND_OPENAI_API_KEY']) if(!bindings.envs?.some(e=>e.key===name&&e.gitBranch===branch&&e.target?.includes('preview'))) throw Error('The authorized Preview branch secret is missing: '+name);
  const manifest=JSON.parse(await readFile(app+'private/templates-manifest.json','utf8'));
  const count=await verifyPrivateTemplateArchives(path.resolve(app,'private'),manifest);
  if(count!==32)throw Error('Preserve all 8 current and 24 retained archives.');
  const selected=new Set(git('ls-files','--',app).split('\n').filter(Boolean));
  selected.add(app+'private/templates-manifest.json');
  for(const item of manifest.archives) selected.add(app+'private/'+item.archiveKey);
  const files=[],inventory=[];
  for(const filename of [...selected].sort()) {
    if(filename.split('/').some(part=>part.startsWith('.env')||['.next','.vercel','node_modules','tests','docs','artifacts'].includes(part)) || /\.(log|tsbuildinfo)$/.test(filename))continue;
    if(!(await lstat(filename)).isFile()||(await lstat(filename)).isSymbolicLink())throw Error('Unsafe source bundle path.');
    const bytes=await readFile(filename);files.push({file:filename,data:bytes.toString('base64'),encoding:'base64'});inventory.push({file:filename,bytes:bytes.length,sha256:digest(bytes)});
  }
  const bundleHash=digest(JSON.stringify(inventory)), sourceRevision=sha+'-bundle-'+bundleHash.slice(0,16);
  Object.assign(env,{EIDOS_SOURCE_COMMIT:sha,EIDOS_SOURCE_BUNDLE_HASH:bundleHash,EIDOS_TEMPLATE_SOURCE_REVISION:sourceRevision});
  const receipt={timestampUtc:new Date().toISOString(),backendSha:sha,bundleHash,sourceRevision,branch,siteOrigin:testSite,mode:'test',
    database:'existing approved TEST validation',archiveCount:count,accounts:true,commerce:'Stripe TEST',mail:'inherited Preview secret',
    playgroundAi:false,inputCountCalls:0,production:false,branchEnvironmentModified:false,privateSecretsPrinted:false,deployed:false};
  await mkdir(output,{recursive:true});await writeFile(output+'/test-backend-inventory.json',JSON.stringify({backendSha:sha,bundleHash,files:inventory},null,2)+'\n');
  if(process.argv.includes('--deploy')) {
    const stripe=async(endpoint,method='GET',body)=>{
      const r=await fetch('https://api.stripe.com/v1/'+endpoint,{method,redirect:'error',signal:AbortSignal.timeout(15000),headers:{authorization:'Bearer '+env.STRIPE_SECRET_KEY,'stripe-version':'2025-06-30.basil',...(body?{'content-type':'application/x-www-form-urlencoded','idempotency-key':'eidos-playground-test-webhook-20261015'}:{})},...(body?{body:new URLSearchParams(body)}:{})});
      const value=await r.json();if(!r.ok)throw Error('Stripe TEST setup failed ('+r.status+').');return value;
    };
    const account=await stripe('account');if(account.id!==merchant||!account.charges_enabled||!account.payouts_enabled)throw Error('The verified TEST merchant is unavailable.');
    let stored;
    try {stored=JSON.parse(await readFile(privatePath,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
    if(stored) {
      const endpoint=await stripe('webhook_endpoints/'+stored.endpointId);
      if(endpoint.livemode!==false||endpoint.url!==testSite+'/api/shop/webhook'||endpoint.status!=='enabled'||!stored.webhookSecret?.startsWith('whsec_'))throw Error('Stored TEST endpoint differs from its approved candidate.');
    } else {
      const data={url:testSite+'/api/shop/webhook',api_version:'2025-06-30.basil',description:'Isolated Eidos Playground TEST acceptance 2026-10-15'};
      ['checkout.session.completed','checkout.session.async_payment_succeeded','checkout.session.async_payment_failed','checkout.session.expired','charge.refunded','charge.dispute.created'].forEach((name,i)=>data['enabled_events['+i+']']=name);
      const endpoint=await stripe('webhook_endpoints','POST',data);
      if(endpoint.livemode!==false||endpoint.url!==testSite+'/api/shop/webhook'||!endpoint.secret?.startsWith('whsec_'))throw Error('Unexpected TEST endpoint result.');
      stored={endpointId:endpoint.id,webhookSecret:endpoint.secret,siteOrigin:testSite};
      await mkdir(path.dirname(privatePath),{recursive:true});await writeFile(privatePath,JSON.stringify(stored),{mode:0o600,flag:'wx'});
    }
    env.EIDOS_KIT_WEBHOOK_SECRET=stored.webhookSecret;
    const payload={name:'eidos-sentinel-lab',project,files,env,gitMetadata:{remoteUrl:'https://github.com/bmparent/eidos',commitRef:branch,commitSha:sha,dirty:false},
      meta:{paidTemplatesGitSha:sha,paidTemplatesSourceBranch:branch,paidTemplatesEnvironmentBranch:branch,paidTemplatesSourceSha256:bundleHash,playgroundTestCandidate:'true'},projectSettings:{rootDirectory:app.slice(0,-1)}};
    const r=vercelApi(args.cli,'/v13/deployments?forceNew=1&teamId='+team,'POST',payload);
    if(r.target==='production')throw Error('Unexpected production target; inspect immediately.');
    Object.assign(receipt,{deployed:true,stripeEndpoint:stored.endpointId,deployment:{id:r.id,url:'https://'+r.url,state:r.readyState||r.status,target:r.target||'preview'}});
  }
  await writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) main().catch(()=>{console.error('Isolated TEST deployment failed; private details suppressed.');process.exitCode=1;});
