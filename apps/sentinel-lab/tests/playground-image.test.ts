import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {validatePlaygroundImage} from '../lib/works/playgroundImage';
test('real raster decoder validates transparent images and rejects forged, truncated, oversized input',async()=>{
 const png=await sharp({create:{width:8,height:8,channels:4,background:{r:10,g:100,b:200,alpha:0.5}}}).png().toBuffer();
 await validatePlaygroundImage('data:image/png;base64,'+png.toString('base64'));
 await assert.rejects(()=>validatePlaygroundImage('data:image/jpeg;base64,'+png.toString('base64')));
 await assert.rejects(()=>validatePlaygroundImage('data:image/png;base64,'+png.subarray(0,40).toString('base64')));
 await assert.rejects(()=>validatePlaygroundImage('data:image/svg+xml;base64,'+Buffer.from('<svg/>').toString('base64')));
 const large=await sharp({create:{width:4001,height:4000,channels:3,background:'#ffffff'}}).png().toBuffer();
 await assert.rejects(()=>validatePlaygroundImage('data:image/png;base64,'+large.toString('base64')));
});
