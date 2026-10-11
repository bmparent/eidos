import { readFile, mkdir, writeFile } from 'node:fs/promises';
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
  const key=`templates/${match[1]}/1.0.0/${match[2]}.zip`, target=path.resolve(targetRoot,key);
  if (!target.startsWith(targetRoot+path.sep)) throw Error('Invalid private archive target');
  let existing;
  try {existing = await readFile(target);} catch (error) {if(error.code!=='ENOENT')throw error;}
  if (existing && createHash('sha256').update(existing).digest('hex') !== digest) throw Error('Refusing to overwrite an immutable version. Publish a new version: '+key);
  await mkdir(path.dirname(target),{recursive:true});
  if (!existing) await writeFile(target,bytes,{flag:'wx'});
  receipts.push({productId:item.editionId,archiveKey:key,sha256:digest,bytes:bytes.length,alreadyPresent:Boolean(existing)});
}
if(receipts.length!==8)throw Error('Expected exactly eight buyer packages');
const output=path.resolve(options.out);await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify({schemaVersion:1,timestampUtc:new Date().toISOString(),sourceCatalogVersion:report.catalogVersion,privateArchives:receipts},null,2)+'\n');
console.log(JSON.stringify({privateArchives:receipts.length,allHashesMatch:true,receipt:options.out}));
