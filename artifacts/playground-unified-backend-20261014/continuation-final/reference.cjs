// Current-source reference fixture. This builder is not the tested Next runtime.
const { createRequire } = require('node:module');
const fs = require('node:fs');
const path = require('node:path');
const frontend = process.argv[2];
if (!frontend) throw Error('Supply the authoritative frontend repository root.');
const { build } = createRequire(path.resolve(frontend, 'package.json'))('esbuild');
(async () => {
  const code = await build({ stdin: { contents: 'export {buildFrozenPackage,createProjectSnapshot} from "./src/playground/purchaseSnapshot"; export {createStarter} from "./src/playground/starters"; export {flattenNodes} from "./src/playground/authoringSchema";', resolveDir: path.resolve(frontend) }, bundle: true, format: 'esm', platform: 'node', write: false, keepNames: true, minify: false });
  const source = await import('data:text/javascript;base64,' + Buffer.from(code.outputFiles[0].text).toString('base64'));
  const pixel = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aLTsAAAAASUVORK5CYII=';
  const project = source.createStarter({starter:'matter',goal:'portfolio',action:'contact',name:'Current compiled fixture'},pixel);
  for (const section of project.sections) if(section.authoring) source.flattenNodes(section.authoring.root).forEach((node,index)=>{node.id='node-compiled-'+section.id+'-'+index;});
  const hero = project.sections.find(section=>section.id==='hero');
  const heading = source.flattenNodes(hero.authoring.root).find(node=>node.type==='heading');
  const image = source.flattenNodes(hero.authoring.root).find(node=>node.type==='image');
  heading.motion={preset:'rise',duration:600,delay:80};
  image.mobile={width:100,height:260};
  const snapshot=await source.createProjectSnapshot(project,'compiled-current-20261014',1);
  const frozen=await source.buildFrozenPackage(snapshot);
  const out='artifacts/playground-unified-backend-20261014/continuation-final/current-reference';
  fs.mkdirSync(out,{recursive:true});
  fs.writeFileSync(out+'/reference.zip',frozen.archive);
  fs.writeFileSync(out+'/snapshot.json',JSON.stringify(snapshot,null,2)+'\n');
  const receipt={timestampUtc:new Date().toISOString(),source:'Current authoritative frontend purchaseSnapshot/starters/common renderer',fixture:'Fresh public Matter starter, deterministic IDs, synthetic one-pixel PNG, block Motion and mobile image sizing',snapshotHash:snapshot.snapshotHash,archiveSha256:frozen.receipt.archiveSha256,archiveBytes:frozen.archive.length,build:'esbuild node ESM keepNames=true minify=false; reference only',actualNextExecution:'Separate plain Node compiled probe required',browserAcceptance:'not asserted',providerCalls:0};
  fs.writeFileSync(out+'/reference-receipt.json',JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify(receipt));
})().catch(error=>{console.error(error);process.exitCode=1;});
