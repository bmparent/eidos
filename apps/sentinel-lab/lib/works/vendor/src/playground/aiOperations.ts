import {type Project,type Section,sectionKind,validateProject,safeHref} from './model';
export const operationFields=['section.title','section.description','section.cta','section.href','section.layout','section.spacing','section.align','tokens.background','tokens.foreground','tokens.accent','tokens.font','tokens.spacing','tokens.typeScale'] as const;
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
export function applyProposal(p:Project,selected:string,scope:'section'|'page',value:unknown):Project {
 const proposal=validateProposal(value),next=structuredClone(p),s=next.sections.find(s=>s.id===selected);if(!s)throw Error('The selected section is gone.');
 const locks=p.brand?.locks||[];
 for(const {field,value} of proposal.ops){
  if(field.startsWith('tokens.')){
   if(scope!=='page')throw Error('Page settings require explicit page scope.');
   const key=field.slice(7) as keyof Project['tokens'];if((['background','foreground','accent'].includes(key)&&locks.includes('palette'))||(key==='font'&&locks.includes('font')))throw Error('The proposed change touches a locked brand value.');
   Object.assign(next.tokens,{[key]:['spacing','typeScale'].includes(key)?Number(value):value});
  }else{
   const key=field.slice(8);if(key==='title'&&['header','footer'].includes(sectionKind(s))&&locks.includes('name'))throw Error('The brand name is locked.');
   if(s.composition&&['layout','spacing','align'].includes(key))throw Error('Use the spatial layout controls for this hero; legacy AI layout fields cannot change it.');
   if(key==='href'&&value&&safeHref(value)==='#'&&value!=='#')throw Error('AI proposed an unsafe link.');
   if(key==='spacing'||key==='align'){if(p.schemaVersion===1||['header','footer'].includes(sectionKind(s)))throw Error('Section layout is unavailable here.');s.style={spacing:p.tokens.spacing,align:'left',background:'',...s.style,[key]:key==='spacing'?Number(value):value} as Section['style'];}
   else Object.assign(s,{[key]:value});
  }
 }
 return validateProject(next);
}
export async function stateHash(p:Project){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(p)))),b=>b.toString(16).padStart(2,'0')).join('');}
export function proposalDiff(p:Project,selected:string,scope:'section'|'page',proposal:Proposal){const next=applyProposal(p,selected,scope,proposal);const a=p.sections.find(s=>s.id===selected)!,b=next.sections.find(s=>s.id===selected)!;return proposal.ops.map(op=>{const [group,key]=op.field.split('.');const before=group==='tokens'?p.tokens[key as keyof Project['tokens']]:key==='spacing'||key==='align'?a.style?.[key]:a[key as keyof Section];const after=group==='tokens'?next.tokens[key as keyof Project['tokens']]:key==='spacing'||key==='align'?b.style?.[key]:b[key as keyof Section];return {field:op.field,before:String(before??'Inherited'),after:String(after??'')};});}
