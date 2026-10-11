import { randomBytes } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { vercelApi } from './vercel-preview-client.mjs';

const options=Object.fromEntries(process.argv.slice(2).filter(v=>v.includes('=')).map(v=>v.replace(/^--/,'').split('=')));
if(!process.argv.includes('--apply')||!options.cli||!options.site||!options['expected-db-host'])throw Error('Explicit --apply, --cli=..., --site=... and --expected-db-host=... are required.');
const site=new URL(options.site);
if(site.protocol!=='https:'||site.hostname!=='eidosworks-templates-test-20261010.pages.dev')throw Error('Unexpected candidate Pages origin.');
const dbUrl=new URL(process.env.EIDOS_DATABASE_URL||'https://invalid');
if(dbUrl.hostname!==options['expected-db-host']||!dbUrl.hostname.includes('validation')||!process.env.EIDOS_DATABASE_AUTH_TOKEN)throw Error('Confirmed validation database required.');
const stripeKey=process.env.STRIPE_SECRET_KEY||'';
if(!/^[sr]k_test_/.test(stripeKey))throw Error('Stripe TEST key required.');
const accountResponse=await fetch('https://api.stripe.com/v1/account',{headers:{authorization:'Bearer '+stripeKey}}),account=await accountResponse.json();
if(!accountResponse.ok||account.id!=='acct_1Rj03gKC8pRG5Tr9')throw Error('Unexpected merchant; no provider mutations made.');
const project='prj_5aL0COF1DONfPKI4fv25NitAdkOE',team='team_TEsnMvKU6SdbYjVhws6Ttp8H';
if(!process.argv.includes('--connector-provider-confirmed')) {
  const provider=vercelApi(options.cli,`/v9/projects/${project}?teamId=${team}`);
  if(provider.id!==project||provider.accountId!==team)throw Error('Unexpected Vercel project/team.');
}
const root=path.resolve('.vercel');await mkdir(root,{recursive:true});
const secretFile=path.join(root,'templates-preview.json');
let previous;
try{previous=JSON.parse(await readFile(secretFile,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
const env={EIDOS_DATABASE_BINDING_PREFIX:'EIDOS_TEMPLATES_TEST',EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL:dbUrl.href,EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN:process.env.EIDOS_DATABASE_AUTH_TOKEN,
  EIDOS_PLATFORM_TOKEN:previous?.env?.EIDOS_PLATFORM_TOKEN||randomBytes(32).toString('hex'),EIDOS_RATE_SECRET:previous?.env?.EIDOS_RATE_SECRET||randomBytes(32).toString('hex'),
  STRIPE_SECRET_KEY:stripeKey,PUBLIC_SITE_URL:site.origin,EIDOS_PREVIEW_ORIGINS:site.origin,EIDOS_SHOP_ENABLED:'true',EIDOS_TEMPLATE_TEST_ENABLED:'true',EIDOS_TEMPLATE_TEST_CHALLENGE:'true',EIDOS_TEMPLATE_VERIFIED_EDITIONS:'',
  EIDOS_ACCOUNTS_ENABLED:'false',EIDOS_AI_ENABLED:'false',EIDOS_NEWSLETTER_ENABLED:'false',EIDOS_PROACTIVE_ENABLED:'false',EIDOS_COMMUNITY_STUDIO_ENABLED:'false',GA_MEASUREMENT_ID:'',
  TURNSTILE_SITE_KEY:'1x00000000000000000000AA',TURNSTILE_SECRET_KEY:'1x0000000000000000000000000000000AA',EIDOS_MAIL_DAILY_LIMIT:'30'};
env.EIDOS_TEMPLATE_VERIFIED_EDITIONS=['switchboard','tideglass','matter','nightjar'].map(slug=>slug+'-developer-v1').join(',');
if(previous?.env?.EIDOS_MAIL_FROM)env.EIDOS_MAIL_FROM=previous.env.EIDOS_MAIL_FROM;
const blocked=[];
for(const [key,id] of [['EIDOS_MAIL_FROM','6i4zZvrc8VKn2Hhp'],['RESEND_API_KEY','eoUrjHqk5m9hqeaP']]){
  if(env[key])continue;
  if(process.argv.includes('--connector-provider-confirmed')){blocked.push(key+' sensitive provider value cannot be read back');continue;}
  try{
    const entry=vercelApi(options.cli,`/v1/projects/${project}/env/${id}?teamId=${team}`);
    if(entry.key!==key||typeof entry.value!=='string')throw Error('Selected environment value unavailable.');
    if(key==='RESEND_API_KEY'&&!/^re_/.test(entry.value))throw Error('Provider did not supply a valid Resend key.');
    env[key]=entry.value;
  }catch{blocked.push(key+' unavailable from selected authorized provider record');}
}
if(env.RESEND_API_KEY){
  const response=await fetch('https://api.resend.com/domains',{headers:{authorization:'Bearer '+env.RESEND_API_KEY}});
  if(!response.ok&&response.status!==403){delete env.RESEND_API_KEY;blocked.push('Resend key failed provider validation');}
}
let bypass=previous?.bypass;
if(!bypass){
  bypass=randomBytes(32).toString('hex');
  const result=vercelApi(options.cli,`/v1/projects/${project}/protection-bypass?teamId=${team}`,'PATCH',{generate:{secret:bypass,note:'Paid templates TEST candidate relay 2026-10-10'}});
  if(!result.protectionBypass?.[bypass])throw Error('Automation bypass creation could not be confirmed.');
}
let webhookId=previous?.webhookId,webhookSecret=previous?.env?.EIDOS_KIT_WEBHOOK_SECRET;
if(!webhookId||!webhookSecret){
  const body=new URLSearchParams({url:site.origin+'/api/shop/webhook',api_version:'2025-06-30.basil',description:'Paid templates isolated TEST candidate 2026-10-10'});
  ['checkout.session.completed','checkout.session.async_payment_succeeded','checkout.session.async_payment_failed','checkout.session.expired','charge.refunded','charge.dispute.created'].forEach((type,index)=>body.set(`enabled_events[${index}]`,type));
  const response=await fetch('https://api.stripe.com/v1/webhook_endpoints',{method:'POST',headers:{authorization:'Bearer '+stripeKey,'content-type':'application/x-www-form-urlencoded','idempotency-key':'eidos-templates-test-hook-20261010'},body});
  const result=await response.json();
  if(!response.ok||result.livemode!==false||!/^whsec_/.test(result.secret||''))throw Error('Stripe TEST webhook endpoint creation failed.');
  webhookId=result.id;webhookSecret=result.secret;
}
env.EIDOS_KIT_WEBHOOK_SECRET=webhookSecret;
await writeFile(secretFile,JSON.stringify({env,bypass,webhookId},null,2)+'\n',{mode:0o600});
for(const [name,value] of [['EIDOS_TEMPLATES_PLATFORM_TOKEN',env.EIDOS_PLATFORM_TOKEN],['EIDOS_TEMPLATES_PREVIEW_BYPASS',bypass]]){
  const result=spawnSync('C:/Program Files/GitHub CLI/gh.exe',['secret','set',name,'--repo','bmparent/brent-parent-intelligence-studio'],{encoding:'utf8',input:value,timeout:60000});
  if(result.status!==0)throw Error('Could not configure candidate-only GitHub secret '+name);
}
const receipt={timestampUtc:new Date().toISOString(),merchantId:account.id,stripeMode:'test',project,team,siteOrigin:site.origin,databaseHost:dbUrl.hostname,webhookId,existingWebhooksChanged:false,productionChanged:false,
  automationBypass:'new additive secret; protected preview retained',turnstile:'Official public always-pass TEST widget and secret; no remote local-test shortcut',githubSecrets:['EIDOS_TEMPLATES_PLATFORM_TOKEN','EIDOS_TEMPLATES_PREVIEW_BYPASS'],mailConfigured:Boolean(env.RESEND_API_KEY&&env.EIDOS_MAIL_FROM),blocked,privateConfiguration:'.vercel/templates-preview.json'};
await mkdir('../../artifacts/paid-templates-20261010',{recursive:true});await writeFile('../../artifacts/paid-templates-20261010/preview-configuration.json',JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));
