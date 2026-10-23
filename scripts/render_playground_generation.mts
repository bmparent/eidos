/** Render the actual saved provider proposal without another API call. ESM entry point. */
import { readFile,mkdir,writeFile } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { ownerCountRequest } from '../apps/sentinel-lab/lib/works/playground-count-probe';
import { applyContextProposal,contextProposalDiff } from '../apps/sentinel-lab/lib/works/project-delivery/src/playground/aiOperations';
import { exportFiles,zipFiles } from '../apps/sentinel-lab/lib/works/project-delivery/src/playground/export';
import { pageDocument } from '../apps/sentinel-lab/lib/works/project-delivery/src/playground/renderer';
const arg=process.argv.find(s=>s.startsWith('--out='))?.slice(6);
if(!arg)throw Error('Supply a versioned --out evidence directory.');
const out=path.resolve(arg),root=path.resolve('artifacts/playground-authenticated-20261015');
if(!out.startsWith(root+path.sep))throw Error('Use a dedicated public diagnostic artifact directory.');
const saved=JSON.parse(await readFile(path.join(out,'generation-ledger-readback.json'),'utf8'));
assert.equal(saved.passed,true);assert.equal(saved.generation.state,'generated');
const built=ownerCountRequest(),before=built.context.project,after=applyContextProposal(built.context,saved.generation.proposal);
const diff=contextProposalDiff(built.context,saved.generation.proposal);
assert.equal(diff.length,1);assert.equal(saved.generation.proposal.ops[0].field,'section.title');
assert.deepEqual(after.sections.map(s=>s.id),before.sections.map(s=>s.id));assert.deepEqual(after.tokens,before.tokens);
assert.equal(after.sections.find(s=>s.id===built.context.sectionId)!.title,saved.generation.proposal.ops[0].value);
const files=exportFiles(after),directory=path.join(out,'rendered');await mkdir(directory,{recursive:true});
for(const file of files){assert.ok(!file.name.includes('..')&&!path.isAbsolute(file.name));const target=path.join(directory,file.name);await mkdir(path.dirname(target),{recursive:true});await writeFile(target,file.data);}
await writeFile(path.join(out,'before.html'),pageDocument(before));await writeFile(path.join(out,'after.html'),pageDocument(after));
const zip=zipFiles(files);await writeFile(path.join(out,'compiled-server-approved.zip'),zip);
const sha=(b:Uint8Array|string)=>createHash('sha256').update(b).digest('hex');
const receipt={timestampUtc:new Date().toISOString(),providerCalls:0,providerRequestId:saved.generation.providerRequestId,generatedProposal:saved.generation.proposal,diff,
  originalProjectSha256:sha(JSON.stringify(before)),appliedProjectSha256:sha(JSON.stringify(after)),stableSectionIds:true,sharedBrandPreserved:true,
  zipBytes:zip.length,zipSha256:sha(zip),files:files.map(f=>({path:f.name,bytes:f.data.length,sha256:sha(f.data)})),
  customerFundingTested:false,productionDeployed:false,passed:true};
await writeFile(path.join(out,'actual-proposal-render.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt));
