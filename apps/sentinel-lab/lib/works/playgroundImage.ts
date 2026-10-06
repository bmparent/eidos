import sharp from 'sharp';
export async function validatePlaygroundImage(data: string): Promise<void> {
 const match=/^data:image\/(png|jpeg|webp);base64,([A-Za-z0-9+/]+=*)$/.exec(data);
 if(!match)throw Error('Unsupported raster image.');
 const bytes=Buffer.from(match[2],'base64');
 if(bytes.length>1_350_000||bytes.toString('base64')!==match[2])throw Error('Image payload limit.');
 const decoder=sharp(bytes,{failOn:'warning',limitInputPixels:16_000_000,limitInputChannels:4,animated:false});
 const meta=await decoder.metadata();
 if(meta.format!==match[1]||!meta.width||!meta.height||meta.width>12000||meta.height>12000||(meta.pages||1)>1)throw Error('Invalid image dimensions or type.');
 // Decode every pixel, not just metadata or the supplied MIME signature.
 await decoder.timeout({seconds:5}).raw().toBuffer();
}
