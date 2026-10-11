import { execFileSync } from 'node:child_process';
import { readFile, lstat, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { vercelApi } from './vercel-preview-client.mjs';

const options=Object.fromEntries(process.argv.slice(2).filter(v=>v.includes('=')).map(v=>v.replace(/^--/,'').split('=')));
if(!options.cli)throw Error('Supply --cli=<authenticated Vercel CLI path>. Dry-run is default; add --deploy for preview only.');
const root=process.cwd(),app='apps/sentinel-lab/';
const git=(...args)=>execFileSync('git',args,{encoding:'utf8',cwd:root}).trim();
const branch=git('branch','--show-current'),commit=git('rev-parse','HEAD');
if(branch!=='codex/paid-templates-20261010')throw Error('Refusing a branch outside this isolated candidate.');
if(JSON.parse(await readFile(app+'package.json','utf8')).name!=='eidos-sentinel-lab')throw Error('Run from repository root.');
const selected=new Set(git('ls-files','-co','--exclude-standard','--',app).split('\n').filter(Boolean));
const archives=JSON.parse(await readFile('artifacts/paid-templates-20261010/private-archive-receipt-final.json','utf8')).privateArchives;
for(const item of archives)selected.add(app+'private/'+item.archiveKey);
const files=[],manifest=[];
for(const file of [...selected].sort()){
  if(!file.startsWith(app))throw Error('Unexpected source scope.');
  if(file.split('/').some(part=>part.startsWith('.env')||['.next','.vercel','node_modules','tests','docs','artifacts','generated_images'].includes(part))||/\.(log|tsbuildinfo)$/.test(file))continue;
  const target=path.resolve(root,file);
  if(!target.startsWith(root+path.sep)||(await lstat(target)).isSymbolicLink())throw Error('Unsafe source path.');
  const data=await readFile(target),sha256=createHash('sha256').update(data).digest('hex');
  const archive=archives.find(item=>file===app+'private/'+item.archiveKey);
  if(archive&&(sha256!==archive.sha256||data.length!==archive.bytes))throw Error('Private archive changed after receipt freeze: '+archive.productId);
  files.push({file,data:data.toString('base64'),encoding:'base64'});manifest.push({file,bytes:data.length,sha256});
}
if(manifest.filter(item=>item.file.startsWith(app+'private/templates/')).length!==8)throw Error('Expected eight private buyer archives.');
const bundleHash=createHash('sha256').update(JSON.stringify(manifest)).digest('hex');
const receipt={timestampUtc:new Date().toISOString(),scope:'protected preview only',branch,commit,bundleHash,dirty:Boolean(git('status','--porcelain')),files:manifest,uploaded:false};
if(process.argv.includes('--deploy')||options['request-out']){
  const configuration=JSON.parse(await readFile(app+'.vercel/templates-preview.json','utf8'));
  if(configuration.env.PUBLIC_SITE_URL!=='https://eidosworks-templates-test-20261010.pages.dev'||configuration.env.EIDOS_DATABASE_BINDING_PREFIX!=='EIDOS_TEMPLATES_TEST'||!/^[sr]k_test_/.test(configuration.env.STRIPE_SECRET_KEY||''))throw Error('Unexpected candidate configuration.');
  const sourceRevision=commit+'-bundle-'+bundleHash.slice(0,16);
  const payload={
    name:'eidos-sentinel-lab',project:'prj_5aL0COF1DONfPKI4fv25NitAdkOE',files,
    gitMetadata:{remoteUrl:'https://github.com/bmparent/eidos',commitRef:branch,commitSha:commit,dirty:receipt.dirty},
    env:{...configuration.env,EIDOS_TEMPLATE_SOURCE_REVISION:sourceRevision},
    meta:{paidTemplatesSourceSha256:bundleHash,paidTemplatesGitSha:commit},
    projectSettings:{rootDirectory:app.slice(0,-1)},
  };
  if(options['request-out']) {
    const output=path.resolve(options['request-out']);
    if(!output.startsWith('E:\\CodexArtifacts\\paid-templates-20261010\\'))throw Error('Private connector request must remain in the explicit external task output directory.');
    await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify(payload),{mode:0o600});
    receipt.sourceRevision=sourceRevision;
  } else {
  const result=vercelApi(options.cli,'/v13/deployments?forceNew=1&teamId=team_TEsnMvKU6SdbYjVhws6Ttp8H','POST',payload);
  if(result.target==='production')throw Error('Unexpected production deployment returned. Inspect provider immediately.');
  Object.assign(receipt,{uploaded:true,sourceRevision,deployment:{id:result.id,url:'https://'+result.url,state:result.readyState||result.status,target:result.target||'preview'}});
  }
}
await mkdir('artifacts/paid-templates-20261010',{recursive:true});await writeFile('artifacts/paid-templates-20261010/backend-source-preview.json',JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({uploaded:receipt.uploaded,bundleHash,files:manifest.length,deployment:receipt.deployment,receipt:'artifacts/paid-templates-20261010/backend-source-preview.json'}));
