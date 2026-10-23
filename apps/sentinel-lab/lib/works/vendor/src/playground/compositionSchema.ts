/** Portable layout data. No CSS strings, remote URLs or arbitrary executable content. */
export const compositionParts = ['title', 'description', 'cta', 'image'] as const;
export type CompositionPart = typeof compositionParts[number];
export const imagePlacements = ['background', 'left', 'right', 'above', 'below', 'inline'] as const;
export type ImagePlacement = typeof imagePlacements[number];
export type Composition = {
  version: 1;
  placement: ImagePlacement;
  mobilePlacement: 'auto' | 'background' | 'above' | 'below' | 'inline';
  order: CompositionPart[];
  align: 'left' | 'center' | 'right';
  mobileAlign: 'inherit' | 'left' | 'center' | 'right';
  gap: number;
  contentWidth: number;
  minHeight: number;
  imageOpacity: number;
  overlay: string;
  overlayOpacity: number;
  blur: number;
  offsetX: number;
  offsetY: number;
  rotation: number;
  scale: number;
};
export function defaultComposition(layout: 'split' | 'center' = 'split'): Composition {
  return {
    version: 1, placement: layout === 'center' ? 'below' : 'right', mobilePlacement: 'auto',
    order: [...compositionParts], align: layout === 'center' ? 'center' : 'left', mobileAlign: 'inherit',
    gap: 24, contentWidth: 100, minHeight: 510, imageOpacity: 1,
    overlay: '#000000', overlayOpacity: 0.35, blur: 0,
    offsetX: 0, offsetY: 0, rotation: 0, scale: 1,
  };
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Invalid composition.');
  return value as Record<string, unknown>;
}
function choice<T extends string>(value: unknown, choices: readonly T[]): T {
  if (typeof value !== 'string' || !choices.includes(value as T)) throw Error('Unsupported composition option.');
  return value as T;
}
function number(value: unknown, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max)
    throw Error(`Composition setting must be between ${min} and ${max}.`);
  return value;
}
export function validateComposition(value: unknown): Composition {
  const c = object(value);
  const keys = Object.keys(defaultComposition());
  if (Object.keys(c).some(key => !keys.includes(key))) throw Error('Unknown composition field.');
  if (c.version !== 1) throw Error('Unsupported composition version.');
  if (!Array.isArray(c.order) || c.order.length !== 4 || new Set(c.order).size !== 4)
    throw Error('Composition must contain each element exactly once.');
  const order = c.order.map(part => choice(part, compositionParts));
  if (typeof c.overlay !== 'string' || !/^#[a-f\d]{6}$/i.test(c.overlay)) throw Error('Invalid overlay color.');
  return {
    version: 1,
    placement: choice(c.placement, imagePlacements),
    mobilePlacement: choice(c.mobilePlacement, ['auto', 'background', 'above', 'below', 'inline']),
    order,
    align: choice(c.align, ['left', 'center', 'right']),
    mobileAlign: choice(c.mobileAlign, ['inherit', 'left', 'center', 'right']),
    gap: number(c.gap, 0, 96), contentWidth: number(c.contentWidth, 30, 100), minHeight: number(c.minHeight, 240, 1000),
    imageOpacity: number(c.imageOpacity, 0.1, 1), overlay: c.overlay, overlayOpacity: number(c.overlayOpacity, 0, 0.85),
    blur: number(c.blur, 0, 20), offsetX: number(c.offsetX, -15, 15), offsetY: number(c.offsetY, -15, 15),
    rotation: number(c.rotation, -15, 15), scale: number(c.scale, 0.75, 1.5),
  };
}
export function effectivePlacement(c: Composition, mobile: boolean): ImagePlacement {
  if (!mobile) return c.placement;
  if (c.mobilePlacement !== 'auto') return c.mobilePlacement;
  return c.placement === 'left' || c.placement === 'right' ? 'below' : c.placement;
}
/** Returns a fresh permutation. It never changes input or removes a content element. */
export function movePart(order: CompositionPart[], part: CompositionPart, before: CompositionPart | null): CompositionPart[] {
  if (!compositionParts.includes(part) || (before !== null && !compositionParts.includes(before)))
    throw Error('Invalid element destination.');
  if (part === before) return [...order];
  const result = order.filter(value => value !== part);
  result.splice(before === null ? result.length : result.indexOf(before), 0, part);
  return result;
}
