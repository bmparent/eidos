import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { vercelApi } from './vercel-preview-client.mjs';

const options = Object.fromEntries(process.argv.slice(2).map(value => value.replace(/^--/, '').split('=')));
if (!options.cli || !options.out) throw Error('Supply --cli=<authenticated CLI path> and --out=<sanitized receipt>');
const project='prj_5aL0COF1DONfPKI4fv25NitAdkOE',team='team_TEsnMvKU6SdbYjVhws6Ttp8H';
const current=vercelApi(options.cli,`/v9/projects/${project}?teamId=${team}`);
if(current.id!==project||current.accountId!==team)throw Error('Unexpected Vercel project/team');
const configuration=vercelApi(options.cli,`/v10/projects/${project}/env?decrypt=false&teamId=${team}`);
const selected=(configuration.envs||[]).filter(item=>['RESEND_API_KEY','EIDOS_MAIL_FROM','EIDOS_MAIL_QA_RECIPIENT'].includes(item.key));
const metadata=selected.map(({id,key,type,target,gitBranch})=>({id,key,type,target,gitBranch:gitBranch||null}));
const receipt={timestampUtc:new Date().toISOString(),project,team,protectedPreview:current.ssoProtection?.deploymentType||null,automationBypassAvailable:Object.values(current.protectionBypass||{}).some(value=>value.scope==='automation'),emailConfiguration:metadata};
const output=path.resolve(options.out);await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));
