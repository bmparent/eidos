import {flattenNodes,isContainer,type BlockNode} from './authoringSchema';
import {type Project,type Section,sectionKind,validateProject,safeHref,labels} from './model';
export const operationFields=['section.title','section.description','section.cta','section.href','section.layout','section.spacing','section.align','tokens.background','tokens.foreground','tokens.accent','tokens.font','tokens.spacing','tokens.typeScale','node.text','node.href','node.alt','node.width','node.height','node.gap','node.align','node.mobile.width','node.mobile.height','node.mobile.gap','node.mobile.align','node.mobile.layout','brand.businessName','brand.tone','page.sectionOrder','page.sectionVisibility'] as const;
export type AiScope='element'|'section'|'page'|'brand';
export type Operation={field:typeof operationFields[number];value:string};
export type Proposal={ops:Operation[]};
export const operationSchema={type:'object',additionalProperties:false,required:['ops'],properties:{ops:{type:'array',minItems:1,maxItems:6,items:{type:'object',additionalProperties:false,required:['field','value'],properties:{field:{type:'string',enum:operationFields},value:{type:'string',maxLength:2000}}}}}};
export function validateProposal(value:unknown):Proposal {
 if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).some(k=>k!=='ops'))throw Error('Invalid AI operation object.');
 const ops=(value as Proposal).ops;if(!Array.isArray(ops)||ops.length<1||ops.length>6)throw Error('AI must propose one to six changes.');
 const fields=new Set<string>();for(const op of ops){if(!op||typeof op!=='object'||Object.keys(op).sort().join(',')!=='field,value'||!operationFields.includes(op.field)||typeof op.value!=='string'||op.value.length>2000||fields.has(op.field))throw Error('Invalid, repeated or unsupported AI field.');fields.add(op.field);}
 return {ops:ops.map(o=>({...o}))};
}
export function aiContext(p:Project,selected:string,scope:'section'|'page') {
 const s=p.sections.find(s=>s.id===selected);if(!s)throw Error('Select a section first.');
 return {schemaVersion:p.schemaVersion,rendererVersion:p.rendererVersion,template:p.template,scope,section:{id:s.id,type:sectionKind(s),title:s.title,description:s.description,cta:s.cta,href:s.href,layout:s.layout,...(s.style?{style:s.style}:{}),...(s.composition?{composition:s.composition}:{})},tokens:p.tokens,brand:p.brand||null};
}
export function applyProposal(p:Project,selected:string,scope:AiScope,value:unknown,selectedNode=''):Project {
 const proposal=validateProposal(value),next=structuredClone(p),s=next.sections.find(s=>s.id===selected);if(!s)throw Error('The selected section is gone.');
 const locks=p.brand?.locks||[];
 for(const {field,value} of proposal.ops){
  if(scope==='element'&&!field.startsWith('node.'))throw Error('Element scope can only change the selected element.');
  if(scope==='brand'&&!field.startsWith('tokens.')&&!field.startsWith('brand.'))throw Error('Brand scope can only change shared brand settings.');
  if(field.startsWith('node.')){
   if(scope!=='element')throw Error('Element settings require explicit element scope.');
   const node=s.authoring&&flattenNodes(s.authoring.root).find(n=>n.id===selectedNode);if(!node)throw Error('The selected element is gone.');
   const key=field.slice(5),mobile=key.startsWith('mobile.'),name=mobile?key.slice(7):key;
   if(name==='text'&&!['heading','text','button'].includes(node.type))throw Error('Only text elements accept copy changes.');
   if(name==='href'&&node.type!=='button')throw Error('Only a button accepts a link change.');
   if(['height','alt'].includes(name)&&node.type!=='image')throw Error('Only an image accepts this setting.');
   if(['gap','layout'].includes(name)&&!isContainer(node))throw Error('Only a container accepts this layout setting.');
   if(name==='href'&&value&&safeHref(value)==='#'&&value!=='#')throw Error('AI proposed an unsafe link.');
   const numeric=['width','height','gap'].includes(name);
   if(numeric&&!/^(?:\d+(?:\.\d+)?|\.\d+)$/.test(value))throw Error('AI geometry must be a finite positive decimal.');
   const patch={[name]:numeric?Number(value):value};
   if(mobile)node.mobile={...node.mobile,...patch};else Object.assign(node,patch);
  }else if(field.startsWith('page.')){
   if(scope!=='page')throw Error('Page structure requires explicit whole-page scope.');
   let parsed:unknown;try{parsed=JSON.parse(value);}catch{throw Error('Page structure must contain valid typed JSON.');}
   if(field==='page.sectionOrder'){
    if(!Array.isArray(parsed)||parsed.length!==next.sections.length||new Set(parsed).size!==parsed.length||parsed.some(id=>typeof id!=='string'||!next.sections.some(section=>section.id===id))||parsed[0]!==next.sections[0].id||parsed.at(-1)!==next.sections.at(-1)!.id)throw Error('Page order must preserve every section and its permanent header/footer.');
    next.sections=(parsed as string[]).map(id=>next.sections.find(section=>section.id===id)!);
   }else{
    if(!parsed||typeof parsed!=='object'||Array.isArray(parsed)||!Object.keys(parsed).length)throw Error('Section visibility must be a bounded object.');
    for(const [id,visible] of Object.entries(parsed)){const section=next.sections.find(section=>section.id===id);if(!section||['header','footer'].includes(sectionKind(section))||typeof visible!=='boolean')throw Error('Only existing content sections accept visibility changes.');section.visible=visible;}
    if(!next.sections.some(section=>section.visible&&!['header','footer'].includes(sectionKind(section))))throw Error('Keep at least one visible content section.');
   }
  }else if(field.startsWith('brand.')){
   if(!['brand','page'].includes(scope))throw Error('Shared brand changes require explicit brand or page scope.');
   if(!next.brand)throw Error('Add a brand with the Brand controls before asking Eidos to change it.');
   const key=field.slice(6) as 'businessName'|'tone';if(locks.includes(key==='businessName'?'name':'tone'))throw Error('The proposed change touches a locked brand value.');
   next.brand[key]=value;
   if(key==='businessName')for(const section of next.sections)if(['header','footer'].includes(sectionKind(section)))section.title=value;
  }else if(field.startsWith('tokens.')){
   if(!['page','brand'].includes(scope))throw Error('Page settings require explicit page or brand scope.');
   const key=field.slice(7) as keyof Project['tokens'];if((['background','foreground','accent'].includes(key)&&locks.includes('palette'))||(key==='font'&&locks.includes('font')))throw Error('The proposed change touches a locked brand value.');
   Object.assign(next.tokens,{[key]:['spacing','typeScale'].includes(key)?Number(value):value});
  }else{
   const key=field.slice(8);if(key==='title'&&['header','footer'].includes(sectionKind(s))&&locks.includes('name'))throw Error('The brand name is locked.');
   if(s.authoring&&key==='layout')throw Error('Use responsive block controls for this layout.');
   if(s.composition&&['layout','spacing','align'].includes(key))throw Error('Use the spatial layout controls for this hero; legacy AI layout fields cannot change it.');
   if(key==='href'&&value&&safeHref(value)==='#'&&value!=='#')throw Error('AI proposed an unsafe link.');
   if(key==='spacing'||key==='align'){if(p.schemaVersion===1||['header','footer'].includes(sectionKind(s)))throw Error('Section layout is unavailable here.');s.style={spacing:p.tokens.spacing,align:'left',background:'',...s.style,[key]:key==='spacing'?Number(value):value} as Section['style'];}
   else {Object.assign(s,{[key]:value});if(s.authoring){const kind=key==='title'?'heading':key==='description'?'text':['cta','href'].includes(key)?'button':null;const node=flattenNodes(s.authoring.root).find(n=>n.type===kind);if(kind&&!node)throw Error('This section has no matching responsive block. Use block controls.');if(node)Object.assign(node,{[key==='href'?'href':'text']:value});}}
  }
 }
 return validateProject(next);
}
export async function stateHash(p:Project){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(p)))),b=>b.toString(16).padStart(2,'0')).join('');}
export function proposalDiff(p:Project,selected:string,scope:AiScope,proposal:Proposal,selectedNode=''){
 const next=applyProposal(p,selected,scope,proposal,selectedNode);
 const read=(project:Project,field:Operation['field'])=>{const section=project.sections.find(s=>s.id===selected)!;const [group,...parts]=field.split('.'),key=parts.join('.');if(group==='page')return project.sections.filter(s=>key!=='sectionVisibility'||s.visible).map(s=>labels[sectionKind(s)]).join(' → ');if(group==='tokens')return project.tokens[key as keyof Project['tokens']];if(group==='brand')return project.brand?.[key as 'businessName'|'tone'];if(group==='node'){const node=section.authoring&&flattenNodes(section.authoring.root).find(n=>n.id===selectedNode);return key.startsWith('mobile.')?node?.mobile?.[key.slice(7) as keyof NonNullable<BlockNode['mobile']>]:node?.[key as keyof BlockNode];}return key==='spacing'||key==='align'?section.style?.[key]:section[key as keyof Section];};
 return proposal.ops.map(op=>({field:op.field,before:String(read(p,op.field)??'Inherited'),after:String(read(next,op.field)??'')}));
}

/** A validated local projection for one proposal. It is not a cloud save or an ownership claim. */
export type ProposalContext={version:1;scope:AiScope;sectionId:string;nodeId:string;project:Project};
export function operationLabel(field:Operation['field']){const [group,...keys]=field.split('.');const name=keys.at(-1)!;const names:Record<string,string>={title:'Heading',description:'Description',cta:'Button text',href:'Link',text:'Text',alt:'Image description',gap:'Spacing',align:'Alignment',width:'Width',height:'Image height',layout:'Layout',font:'Typography',typeScale:'Type scale',businessName:'Business name',tone:'Voice and tone',sectionOrder:'Section order',sectionVisibility:'Visible sections',background:'Background',foreground:'Text color',accent:'Accent color',spacing:'Spacing'};return (keys[0]==='mobile'?'Mobile ':group==='tokens'?'Shared ':'')+(names[name]||name);}
const contextKeys=['version','scope','sectionId','nodeId','project'];
export function proposalTargetId(context:ProposalContext){return context.scope==='element'?context.nodeId:context.scope==='section'?context.sectionId:context.scope;}
export function createProposalContext(input:Project,sectionId:string,nodeId:string,scope:AiScope):ProposalContext {
 const project=structuredClone(validateProject(input));
 const selected=project.sections.find(s=>s.id===sectionId);if(!selected)throw Error('Select a section first.');
 if(['element','section'].includes(scope)&&!selected.visible)throw Error('Show this section with the page controls before asking Eidos to change it.');
 if(scope==='element'&&(!selected.authoring||!flattenNodes(selected.authoring.root).some(n=>n.id===nodeId)))throw Error('Select an editable element first.');
 for(const section of project.sections){
  section.image='';delete section.media;
  if(section.id!==sectionId){section.title=section.title.slice(0,160);section.description=section.description.slice(0,160);}
  for(const card of section.cards||[]){card.image='';delete card.media;card.description=card.description.slice(0,160);}
  if(section.authoring)for(const node of flattenNodes(section.authoring.root)){node.image='';delete node.media;if(scope==='element'&&node.id!==nodeId)node.text=node.text.slice(0,160);}
 }
 if(project.studios)project.studios={version:1,active:'page'};
 return validateProposalContext({version:1,scope,sectionId,nodeId:scope==='element'?nodeId:'',project});
}
export function validateProposalContext(input:unknown):ProposalContext {
 if(!input||typeof input!=='object'||Array.isArray(input))throw Error('Invalid Eidos context.');
 const raw=input as Record<string,unknown>;
 if(Object.keys(raw).sort().join(',')!==contextKeys.slice().sort().join(',')||raw.version!==1||!['element','section','page','brand'].includes(String(raw.scope))||typeof raw.sectionId!=='string'||typeof raw.nodeId!=='string')throw Error('Unsupported Eidos scope or context field.');
 if(JSON.stringify(raw).length>48000)throw Error('This context is too large. Choose a narrower scope or shorten the request.');
 const project=validateProject(raw.project),section=project.sections.find(s=>s.id===raw.sectionId);
 if(!section)throw Error('The Eidos context section is missing.');
 if(['element','section'].includes(String(raw.scope))&&!section.visible)throw Error('Show this section before asking Eidos to change it.');
 if(raw.scope==='element'&&(!section.authoring||!flattenNodes(section.authoring.root).some(n=>n.id===raw.nodeId)))throw Error('The Eidos context element is missing.');
 if(raw.scope!=='element'&&raw.nodeId!=='')throw Error('A broader scope cannot carry an element target.');
 if(project.studios&&(project.studios.active!=='page'||project.studios.artwork||project.studios.motion))throw Error('Do not submit studio assets to Eidos.');
 for(const section of project.sections){if(section.image||section.media)throw Error('Do not submit raw assets to Eidos.');for(const card of section.cards||[])if(card.image||card.media)throw Error('Do not submit raw assets to Eidos.');if(section.authoring)for(const node of flattenNodes(section.authoring.root))if(node.image||node.media)throw Error('Do not submit raw assets to Eidos.');}
 return {version:1,scope:raw.scope as AiScope,sectionId:raw.sectionId,nodeId:raw.nodeId,project};
}
export function applyContextProposal(input:ProposalContext,value:unknown){const context=validateProposalContext(input);const next=applyProposal(context.project,context.sectionId,context.scope,value,context.nodeId);if(JSON.stringify(next)===JSON.stringify(context.project))throw Error('Eidos did not propose a change.');return next;}
export function contextProposalDiff(context:ProposalContext,value:unknown){const proposal=validateProposal(value);applyContextProposal(context,proposal);return proposalDiff(context.project,context.sectionId,context.scope,proposal,context.nodeId);}
