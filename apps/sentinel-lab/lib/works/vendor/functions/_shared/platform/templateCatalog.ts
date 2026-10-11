/** Server-owned TEST candidate prices. These do not alter Cinematic Starter. */
export const TEMPLATE_STRIPE_API_VERSION = '2025-06-30.basil';
export const templateFamilies = ['switchboard', 'tideglass', 'matter', 'nightjar'] as const;
export const templateEditions = ['developer', 'wordpress'] as const;
export type TemplateEdition = typeof templateEditions[number];
export function templateVersion(slug: typeof templateFamilies[number], editionId: TemplateEdition) {
  return editionId === 'wordpress' || slug === 'nightjar' ? '1.0.1' : '1.0.0';
}
export interface TemplateProduct {
  id: string; productId: string; editionId: TemplateEdition; name: string;
  version: string; priceCents: number; archiveKey: string; downloadName: string; guidePath: string;
}
export const templateProducts: TemplateProduct[] = templateFamilies.flatMap(slug => templateEditions.map(editionId => {
  const version = templateVersion(slug, editionId);
  return {
    id: `${slug}-${editionId}-v1`, productId: `${slug}-${editionId}-v1`, editionId,
    name: `${slug[0].toUpperCase()}${slug.slice(1)} — ${editionId === 'wordpress' ? 'WordPress block theme' : 'developer source'}`,
    version, priceCents: editionId === 'developer' ? 9900 : 12900,
    archiveKey: `templates/${slug}/${version}/${editionId}.zip`,
    downloadName: `eidos-${slug}-${editionId}-${version}.zip`,
    guidePath: `/templates/guides/${slug}/${editionId}/`,
  };
}));
export function templateProduct(product: unknown, edition: unknown): TemplateProduct | undefined {
  if (typeof product !== 'string' || typeof edition !== 'string') return;
  return templateProducts.find(item => item.editionId === edition && (item.id === product || item.id === `${product}-${edition}-v1`));
}
