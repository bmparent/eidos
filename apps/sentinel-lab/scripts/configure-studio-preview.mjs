/** Branch-only Preview configuration; credentials remain in existing ignored input. */
import {readFile,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {vercelApi} from './vercel-preview-client.mjs';
const options=Object.fromEntries(process.argv.slice(2).filter(value=>value.includes('=')).map(value=>value.replace(/^--/,'').split('=')));
const branch=execFileSync('git',['branch','--show-current'],{encoding:'utf8'}).trim();
if(branch!=='codex/showcase-studio-boundary-20261013')throw Error('Run from the isolated studio boundary branch.');
if(!options.configuration||!options.cli)throw Error('Supply the existing ignored candidate configuration and authenticated CLI entry.');
const configuration=JSON.parse(await readFile(options.configuration,'utf8'));
const input=configuration.env||{};
if(input.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!new URL(input.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname.includes('validation')||!/^[sr]k_test_/.test(input.STRIPE_SECRET_KEY||''))throw Error('Expected the existing isolated TEST validation configuration.');
const origin='https://eidosworks-showcase-20261013.pages.dev';
const env={...input,PUBLIC_SITE_URL:origin,EIDOS_PREVIEW_ORIGINS:origin,
 EIDOS_SHOP_ENABLED:'false',EIDOS_TEMPLATE_TEST_ENABLED:'false',EIDOS_TEMPLATE_LIVE_ENABLED:'false',
 EIDOS_ACCOUNTS_ENABLED:'false',EIDOS_AI_ENABLED:'false',EIDOS_NEWSLETTER_ENABLED:'false',EIDOS_PROACTIVE_ENABLED:'false',EIDOS_COMMUNITY_STUDIO_ENABLED:'false',
 OPENAI_API_KEY:'',EIDOS_PLAYGROUND_STRIPE_KEY:'',EIDOS_PLAYGROUND_WEBHOOK_SECRET:'',RESEND_API_KEY:''};
const receipt={branch,target:'preview',project:'prj_5aL0COF1DONfPKI4fv25NitAdkOE',team:'team_TEsnMvKU6SdbYjVhws6Ttp8H',siteOrigin:origin,databaseBindingPrefix:env.EIDOS_DATABASE_BINDING_PREFIX,databaseHost:new URL(env.EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL).hostname,providerMode:'TEST key retained; all commerce/account/AI/newsletter/proactive features disabled',configured:false,variableNames:Object.keys(env),timestampUtc:new Date().toISOString()};
if(process.argv.includes('--apply')){
 for(const [key,value]of Object.entries(env)){
  const result=vercelApi(options.cli,'/v10/projects/'+receipt.project+'/env?upsert=true&teamId='+receipt.team,'POST',{key,value,type:/SECRET|TOKEN|KEY/.test(key)?'sensitive':'encrypted',target:['preview'],gitBranch:branch});
  if(!result.created||result.failed?.length)throw Error('New branch Preview configuration failed. Existing branches and Production were not retargeted.');
 }
 receipt.configured=true;
}
await writeFile('artifacts/showcase-studio-boundary-20261013/preview-environment.json',JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));
