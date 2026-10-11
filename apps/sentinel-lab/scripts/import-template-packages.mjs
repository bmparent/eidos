import { readFile, mkdir, readdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

const options = Object.fromEntries(process.argv.slice(2).map(value => value.replace(/^--/, '').split('=')));
if (!options.site || !options.out) throw Error('Supply --site=<website repo root> and --out=<receipt path>');
const site = path.resolve(options.site), targetRoot = path.resolve(new URL('../private/', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const report = JSON.parse(await readFile(path.join(site,'artifacts/templates/package-report.json'),'utf8'));
const receipts = [];
for (const item of report.packages || []) {
  const match = /^(switchboard|tideglass|matter|nightjar)-(developer|wordpress)-v1$/.exec(item.editionId || '');
  if (!match || !/^[a-f0-9]{64}$/.test(item.sha256)) throw Error('Invalid package identity/receipt');
  const source = path.resolve(site,item.packagePath);
  if (!source.startsWith(site+path.sep)) throw Error('Package source escapes website root');
  const bytes = await readFile(source), digest = createHash('sha256').update(bytes).digest('hex');
  if (digest !== item.sha256 || bytes.length !== item.bytes) throw Error('Package differs from frozen source receipt: '+item.editionId);
  const expectedVersion=match[2]==='wordpress'?'1.0.1':match[1]==='nightjar'?'1.0.3':'1.0.0';
  if(item.version!==expectedVersion)throw Error('Package version differs from the server catalog: '+item.editionId);
  const key=`templates/${match[1]}/${item.version}/${match[2]}.zip`, target=path.resolve(targetRoot,key);
  if (!target.startsWith(targetRoot+path.sep)) throw Error('Invalid private archive target');
  let existing;
  try {existing = await readFile(target);} catch (error) {if(error.code!=='ENOENT')throw error;}
  if (existing && createHash('sha256').update(existing).digest('hex') !== digest) throw Error('Refusing to overwrite an immutable version. Publish a new version: '+key);
  await mkdir(path.dirname(target),{recursive:true});
  if (!existing) await writeFile(target,bytes,{flag:'wx'});
  receipts.push({productId:item.editionId,archiveKey:key,sha256:digest,bytes:bytes.length,alreadyPresent:Boolean(existing)});
}
if(receipts.length!==8)throw Error('Expected exactly eight buyer packages');
const retainedArchives=[];
for(const entry of await readdir(path.join(targetRoot,'templates'),{recursive:true,withFileTypes:true})) {
  if(!entry.isFile())continue;
  const target=path.join(entry.parentPath,entry.name),key=path.relative(targetRoot,target).split(path.sep).join('/');
  if(!/^templates\/(switchboard|tideglass|matter|nightjar)\/\d+\.\d+\.\d+\/(developer|wordpress)\.zip$/.test(key))throw Error('Unexpected private archive path');
  if(receipts.some(item=>item.archiveKey===key))continue;
  const bytes=await readFile(target);
  retainedArchives.push({archiveKey:key,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length});
}
const output=path.resolve(options.out);await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify({schemaVersion:2,timestampUtc:new Date().toISOString(),sourceCatalogVersion:report.catalogVersion,privateArchives:receipts,retainedArchives},null,2)+'\n');
console.log(JSON.stringify({privateArchives:receipts.length,retainedArchives:retainedArchives.length,allHashesMatch:true,receipt:options.out}));
