export type MediaSettings = {
 generation?: {model:string;quality:string;providerId:string};
  fit: 'cover' | 'contain' | 'fill'; zoom: number; x: number; y: number;
  mobileX?: number; mobileY?: number; tint: string; opacity: number;
  decorative: boolean; sourceId: string; sourceName: string;
};
export type BrandKit = { businessName: string; tone: string; locks: ('name'|'logo'|'palette'|'font'|'tone')[] };
export const defaultMedia = (): MediaSettings => ({fit:'cover',zoom:1,x:50,y:50,tint:'#000000',opacity:0,decorative:false,sourceId:'',sourceName:''});
function obj(v:unknown):Record<string,unknown> {if(!v || typeof v!=='object' || Array.isArray(v))throw Error('Invalid media settings.');return v as Record<string,unknown>;}
function bounded(v:unknown,min:number,max:number) {if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max)throw Error('Media value outside its supported range.');return v;}
function text(v:unknown,max:number) {if(typeof v!=='string'||v.length>max)throw Error('Invalid media text.');return v;}
export function validateMedia(v:unknown):MediaSettings {
 const m=obj(v);if(!['cover','contain','fill'].includes(String(m.fit))||typeof m.decorative!=='boolean'||!/^#[a-f\d]{6}$/i.test(String(m.tint)))throw Error('Invalid image placement.');
 const sourceId=text(m.sourceId,64);if(sourceId&&!/^[a-f\d]{64}$/.test(sourceId))throw Error('Invalid image identity.');
 const generation=m.generation===undefined?undefined:obj(m.generation);
 return {...(generation?{generation:{model:text(generation.model,80),quality:text(generation.quality,20),providerId:text(generation.providerId,120)}}:{}),fit:m.fit as MediaSettings['fit'],zoom:bounded(m.zoom,1,3),x:bounded(m.x,0,100),y:bounded(m.y,0,100),...(m.mobileX===undefined?{}:{mobileX:bounded(m.mobileX,0,100)}),...(m.mobileY===undefined?{}:{mobileY:bounded(m.mobileY,0,100)}),tint:String(m.tint),opacity:bounded(m.opacity,0,0.8),decorative:m.decorative,sourceId,sourceName:text(m.sourceName,120)};
}
export function validateBrand(v:unknown):BrandKit {
 const b=obj(v);if(!Array.isArray(b.locks)||b.locks.length>5||!b.locks.every(x=>['name','logo','palette','font','tone'].includes(x)))throw Error('Invalid brand locks.');
 return {businessName:text(b.businessName,100),tone:text(b.tone,240),locks:[...new Set(b.locks)] as BrandKit['locks']};
}
