/** Read-only release binding/TEST-provider audit. Never prints secret values. */
import { readFile,mkdir,writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { vercelApi } from '../apps/sentinel-lab/scripts/vercel-preview-client.mjs';
import { adaptDatabase } from '../apps/sentinel-lab/lib/works/database';
import { aiPolicyStatus } from '../apps/sentinel-lab/lib/works/vendor/functions/_shared/platform/playgroundAi';
const args=Object.fromEntries(process.argv.slice(2).filter(s=>s.startsWith('--')&&s.includes('=')).map(s=>{const i=s.indexOf('=');return [s.slice(2,i),s.slice(i+1)];}));
if(!args.configuration||!args.cli||!args.out)throw Error('Supply the approved ignored TEST --configuration, --cli and --out.');
const out=path.resolve(args.out),root=path.resolve('artifacts/playground-authenticated-20261015');
if(!out.startsWith(root+path.sep))throw Error('Use a dedicated versioned artifact directory.');
const {env}=JSON.parse(await readFile(args.configuration,'utf8'));
if(env.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!new URL(env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation'))throw Error('Only the approved TEST database may be audited.');
const require=createRequire(path.resolve('apps/sentinel-lab/package.json')),{createClient}=require('@libsql/client');
const client=createClient({url:env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL,authToken:env.EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN});
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const result:Record<string,unknown>={timestampUtc:new Date().toISOString(),readOnly:true,openaiCalls:0,secretValuesPrinted:false,providerKeyReadLocally:false,
  localBindings:{testDatabase:true,stripeTestKeyPresent:/^[sr]k_test_/.test(env.STRIPE_SECRET_KEY||''),mailKeyPresent:Boolean(env.RESEND_API_KEY),mailFromPresent:Boolean(env.EIDOS_MAIL_FROM),turnstileSitePresent:Boolean(env.TURNSTILE_SITE_KEY)}};
try{
  const configuration=vercelApi(args.cli,'/v10/projects/prj_5aL0COF1DONfPKI4fv25NitAdkOE/env?decrypt=false&teamId=team_TEsnMvKU6SdbYjVhws6Ttp8H');
  const names=['EIDOS_PLAYGROUND_OPENAI_API_KEY','RESEND_API_KEY','EIDOS_MAIL_FROM','EIDOS_MAIL_QA_RECIPIENT','EIDOS_ACCOUNTS_ENABLED','EIDOS_PLAYGROUND_AI_MODE'];
  result.hostedBindingMetadata=(configuration.envs||[]).filter((r:{key:string;gitBranch?:string})=>names.includes(r.key)&&['codex/playground-unified-20261014','codex/paid-templates-20261010'].includes(r.gitBranch||'')).map((r:{key:string;id:string;type:string;target:string[];gitBranch?:string})=>({key:r.key,id:r.id,type:r.type,target:r.target,branch:r.gitBranch||null}));
  result.counts=Object.fromEntries(await Promise.all(['eidos_email_members','eidos_orders','eidos_template_orders','eidos_pg_projects','eidos_pg_orders','eidos_pg_ai_actions','eidos_pg_ai_grants','eidos_pg_ai_paid_receipts'].map(async table=>[table,Number((await client.execute('SELECT COUNT(*) n FROM '+table)).rows[0].n)])));
  result.aiPolicy=await aiPolicyStatus({EIDOS_DB:adaptDatabase(client)},'test');
  result.templateOrderStates=(await client.execute('SELECT o.status,t.fulfillment_status,t.product_id,t.version,COUNT(*) n FROM eidos_orders o JOIN eidos_template_orders t ON t.order_id=o.id GROUP BY o.status,t.fulfillment_status,t.product_id,t.version')).rows;
  const members=(await client.execute('SELECT id,email,disabled FROM eidos_email_members')).rows;
  result.ownedInboxMembers=members.filter(m=>['brent@eidos-works.com','1brent.bm@gmail.com'].includes(String(m.email))).map(m=>({memberIdSha256:sha(String(m.id)),inbox:m.email==='brent@eidos-works.com'?'business':'personal',disabled:Boolean(m.disabled)}));
  const key=env.STRIPE_SECRET_KEY;
  if(!/^[sr]k_test_/.test(key||''))throw Error('An approved Stripe TEST key is required; no fallback is allowed.');
  const stripe=async(endpoint:string)=>{
    const r=await fetch('https://api.stripe.com/v1/'+endpoint,{redirect:'error',signal:AbortSignal.timeout(15000),headers:{authorization:'Bearer '+key}});
    const value=await r.json();if(!r.ok)throw Error('Stripe read rejected: '+r.status);return value;
  };
  const account=await stripe('account');
  result.stripeAccount={id:account.id,merchantMatches:account.id==='acct_1Rj03gKC8pRG5Tr9',configuredMerchantMatches:account.id===env.EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID};
  if(account.id!=='acct_1Rj03gKC8pRG5Tr9')throw Error('Unexpected TEST merchant.');
  const endpoints=await stripe('webhook_endpoints?limit=100'),events=await stripe('events?limit=8');
  if([...endpoints.data,...events.data].some((r:{livemode:boolean})=>r.livemode!==false))throw Error('Provider returned a LIVE resource.');
  result.stripeTestEndpoints=endpoints.data.map(r=>{const u=new URL(r.url);return {id:r.id,url:u.origin+u.pathname,status:r.status,livemode:r.livemode,eventTypes:r.enabled_events};});
  result.recentStripeTestEvents=events.data.map(r=>({id:r.id,type:r.type,created:r.created,livemode:r.livemode,pendingWebhooks:r.pending_webhooks}));
  result.passed=true;
}catch(error){result.passed=false;result.reason=error instanceof Error&&/^(Stripe read rejected: \d+|An approved Stripe TEST key|Unexpected TEST merchant|Provider returned a LIVE resource)/.test(error.message)?error.message:'Read-only audit failed; private details suppressed.';process.exitCode=1;}
finally{client.close();await mkdir(out,{recursive:true});await writeFile(path.join(out,'release-binding-audit.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));}
