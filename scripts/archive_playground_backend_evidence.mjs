import { readdir, readFile, writeFile, mkdir, copyFile, access, lstat } from 'node:fs/promises';
import { constants } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const authenticated=process.argv.includes('--authenticated');
const local=path.resolve(authenticated?'artifacts/playground-authenticated-20261015':'artifacts/playground-unified-backend-20261014'),timestamp=new Date().toISOString();
const dateArg=process.argv.find(value=>value.startsWith('--date='));
const date=dateArg?dateArg.slice(7):timestamp.slice(0,10);
if(!/^\d{4}-\d{2}-\d{2}$/.test(date))throw Error('Use --date=YYYY-MM-DD for the client task date.');
const git=(...args)=>execFileSync('git',args,{encoding:'utf8'}).trim();
const sourceCommit=git('rev-parse','HEAD');
if(git('diff','HEAD','--','apps/sentinel-lab'))throw Error('Application source changed after its committed freeze.');
const sha=data=>createHash('sha256').update(data).digest('hex');
const runId='backend-'+sourceCommit.slice(0,8)+'-'+timestamp.replace(/[:.]/g,'-');
const metadataRelative='archive_receipts/'+runId,metadata=path.join(local,metadataRelative);
const fixtureZips=new Set(['reference.zip','canonical-reference.zip','compiled-server-approved.zip','browser-approved.zip','fixture-input.zip']);
async function files(directory,prefix=''){
  const list=[];
  for(const entry of await readdir(directory,{withFileTypes:true})){
    const relative=path.posix.join(prefix,entry.name),absolute=path.join(directory,entry.name);
    if(entry.isSymbolicLink())throw Error('Evidence symlinks are not archived: '+relative);
    if(entry.isDirectory())list.push(...await files(absolute,relative));
    else if(entry.isFile()){
      if(/(^|\/)(\.env[^/]*|\.vercel|private)(\/|$)/.test(relative))throw Error('Private environment/archive path is outside this evidence contract: '+relative);
      if(relative.endsWith('.zip')&&!fixtureZips.has(entry.name))throw Error('ZIP lacks an approved public-fixture filename: '+relative);
      list.push(relative);
    }
  }
  return list.sort();
}
const sourcePaths=git('ls-files','--','apps/sentinel-lab').split(/\r?\n/).filter(file=>file&&!/(^|\/)(\.env[^/]*|\.vercel|\.next|node_modules|private)(\/|$)/.test(file));
const extraSource=authenticated?['scripts/archive_playground_backend_evidence.mjs','scripts/verify_playground_provider_auth.mjs','docs/playground-authenticated-20261015.md']:['scripts/archive_playground_backend_evidence.mjs','docs/works-playground-continuation-20261014.md'];
sourcePaths.push(...extraSource);
const sourceFiles=[];
for(const file of sourcePaths.sort()){
  if(!file.startsWith('apps/sentinel-lab/')&&!extraSource.includes(file))throw Error('Unexpected source inventory path.');
  const stat=await lstat(file);if(!stat.isFile()||stat.isSymbolicLink())throw Error('Unsafe source inventory file: '+file);
  const data=await readFile(file);sourceFiles.push({path:file,bytes:data.length,sha256:sha(data)});
}
const source={schemaVersion:1,timestampUtc:timestamp,sourceCommit,applicationSourceClean:true,workingTreeChanges:git('status','--porcelain').split(/\r?\n/).filter(Boolean),scope:'Committed app source hashes plus archive helper and current task guide; no private configs, credentials or original paid archives',inventorySha256:sha(JSON.stringify(sourceFiles)),files:sourceFiles};
if(process.argv.includes('--dry-run')){const list=await files(local);console.log(JSON.stringify({dryRun:true,sourceCommit,sourceFiles:sourceFiles.length,artifactFiles:list.length,privateValuesRead:false,copyAttempted:false}));process.exit(0);}
await mkdir(metadata,{recursive:true});
await writeFile(path.join(metadata,'source_inventory.json'),JSON.stringify(source,null,2)+'\n');
const manifest={schemaVersion:1,timestampUtc:timestamp,sourceCommit,sourceInventorySha256:source.inventorySha256,metadataExcluded:[metadataRelative+'/artifact_manifest.json',metadataRelative+'/drive_manifest.json'],files:[]};
for(const relative of await files(local)){if(manifest.metadataExcluded.includes(relative))continue;const data=await readFile(path.join(local,relative));manifest.files.push({path:relative,bytes:data.length,sha256:sha(data)});}
const manifestBytes=Buffer.from(JSON.stringify(manifest,null,2)+'\n');
await writeFile(path.join(metadata,'artifact_manifest.json'),manifestBytes);
const drive={drive_copy_attempted:false,drive_copy_success:false,drive_root:'unknown',drive_run_dir:'unknown',reason:'No configured writable Drive root.',files_considered:manifest.files.map(file=>file.path),files_copied:[],files_skipped:[],timestamp_utc:timestamp,remote_sync_proven:false};
let driveRoot;
for(const candidate of [process.env.EIDOS_PROOF_DRIVE_DIR,process.env.EIDOS_ARTIFACT_ROOT]){if(!candidate)continue;try{await access(candidate,constants.W_OK);driveRoot=path.resolve(candidate);break;}catch{/* Try the next configured destination. */}}
if(driveRoot){
  drive.drive_copy_attempted=true;drive.drive_root=driveRoot;
  const parent=path.join(driveRoot,'Eidos_Works_Playground',date),destination=path.join(parent,runId);drive.drive_run_dir=destination;
  try{
    await mkdir(parent,{recursive:true});await mkdir(destination);
    for(const file of manifest.files){const target=path.join(destination,file.path);await mkdir(path.dirname(target),{recursive:true});await copyFile(path.join(local,file.path),target,constants.COPYFILE_EXCL);if(sha(await readFile(target))!==file.sha256)throw Error('Mirror checksum mismatch: '+file.path);drive.files_copied.push({...file,verified:true});}
    const mirroredMetadata=path.join(destination,metadataRelative);await mkdir(mirroredMetadata,{recursive:true});
    await copyFile(path.join(metadata,'artifact_manifest.json'),path.join(mirroredMetadata,'artifact_manifest.json'),constants.COPYFILE_EXCL);
    if(sha(await readFile(path.join(mirroredMetadata,'artifact_manifest.json')))!==sha(manifestBytes))throw Error('Mirror artifact manifest checksum mismatch.');
    drive.drive_copy_success=true;drive.reason='Every listed artifact and the artifact manifest copied with matching SHA256 readback. Mounted-folder copy does not establish remote cloud synchronization.';
  }catch(error){drive.reason=error.message;drive.files_skipped=manifest.files.filter(file=>!drive.files_copied.some(copied=>copied.path===file.path)).map(file=>({path:file.path,reason:drive.reason}));}
}
let driveBytes=Buffer.from(JSON.stringify(drive,null,2)+'\n');await writeFile(path.join(metadata,'drive_manifest.json'),driveBytes);
let driveMetadataVerified=false;
if(drive.drive_copy_success){try{const mirroredManifest=path.join(drive.drive_run_dir,metadataRelative,'drive_manifest.json');await copyFile(path.join(metadata,'drive_manifest.json'),mirroredManifest,constants.COPYFILE_EXCL);driveMetadataVerified=sha(await readFile(mirroredManifest))===sha(driveBytes);if(!driveMetadataVerified)throw Error('Drive manifest checksum mismatch.');}catch(error){drive.drive_copy_success=false;drive.reason=error.message;driveBytes=Buffer.from(JSON.stringify(drive,null,2)+'\n');await writeFile(path.join(metadata,'drive_manifest.json'),driveBytes);}}
console.log(JSON.stringify({local,sourceCommit,sourceFiles:sourceFiles.length,artifactFiles:manifest.files.length,artifactBytes:manifest.files.reduce((sum,file)=>sum+file.bytes,0),metadataDirectory:metadata,artifactManifestSha256:sha(manifestBytes),driveManifestSha256:sha(driveBytes),drive_copy_success:drive.drive_copy_success,driveMetadataVerified,drive_run_dir:drive.drive_run_dir,reason:drive.reason}));
