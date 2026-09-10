import {body,guarded,json,origin} from '../../_shared/platform/core';
import {requireMember} from '../../_shared/platform/memberAuth';
import {enabledAI,requestAI,resultAI} from '../../_shared/platform/playgroundAI';
export const onRequestGet=guarded(async({request,env})=>{const member=await requireMember(request,env,false),key=new URL(request.url).searchParams.get('requestId');return json(key?await resultAI(env,member.id,key):{ownerId:member.id,enabled:Boolean(await enabledAI(env)),images:env.EIDOS_PLAYGROUND_IMAGE_ENABLED==='true'});});
export const onRequestPost=guarded(async({request,env})=>{origin(request);const member=await requireMember(request,env,false);const input=await body(request,16000);if(input.expectedOwner!==member.id)return json({error:'Your account changed. Local work is preserved.'},409);return json(await requestAI(env,member.id,input));});
