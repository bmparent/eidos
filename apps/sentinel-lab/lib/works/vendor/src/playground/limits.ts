// Total serialized UTF-8 document, including embedded media, before asset extraction.
export const CLOUD_DOCUMENT_BYTES = 2_000_000;
export function cloudPreflight(document: unknown) {
  const bytes = new TextEncoder().encode(JSON.stringify(document)).length;
  const sections = document && typeof document === 'object' && 'sections' in document && Array.isArray(document.sections) ? document.sections as {image?:string;cards?:{image?:string}[]}[] : [];
  const images=sections.flatMap(s=>[s,...(s.cards||[])]).map(s=>typeof s.image==='string'?s.image:'');
  const imageBytes=images.reduce((n,s)=>n+s.length,0);
  const imageError=imageBytes>1_800_000 ? 'This page exceeds its 1.8 MB embedded-image allowance. Use smaller images; local content and exports remain available.' : images.some(s=>s.length>1_800_024) ? 'An image exceeds the 1.35 MB decoded asset limit.' : '';
  return { bytes, allowed: bytes <= CLOUD_DOCUMENT_BYTES && !imageError, message: imageError || (bytes > CLOUD_DOCUMENT_BYTES
    ? `This design uses ${bytes.toLocaleString()} bytes; account saves allow ${CLOUD_DOCUMENT_BYTES.toLocaleString()}. Use smaller images. Your local design and free exports are preserved.` : '') };
}
