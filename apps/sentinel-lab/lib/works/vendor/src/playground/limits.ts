// Total serialized UTF-8 document, including embedded media, before asset extraction.
export const CLOUD_DOCUMENT_BYTES = 2_000_000;
export function cloudPreflight(document: unknown) {
  const bytes = new TextEncoder().encode(JSON.stringify(document)).length;
  return { bytes, allowed: bytes <= CLOUD_DOCUMENT_BYTES, message: bytes > CLOUD_DOCUMENT_BYTES
    ? `This design uses ${bytes.toLocaleString()} bytes; account saves allow ${CLOUD_DOCUMENT_BYTES.toLocaleString()}. Use smaller images. Your local design and free exports are preserved.` : '' };
}
