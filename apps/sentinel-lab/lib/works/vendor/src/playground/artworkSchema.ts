export const ARTWORK_VERSION = 1;
export const ARTWORK_LIMITS = { objects: 48, width: 1600, text: 300, imageBytes: 4_500_000 } as const;
export const artworkKinds = ['bloom', 'leaf', 'orbit', 'square', 'text', 'image'] as const;
export type ArtworkKind = (typeof artworkKinds)[number];
export type PaletteSlot = 'ink' | 'accent' | 'paper';
export type ArtworkObject = {
  id: string; kind: ArtworkKind; name: string; group: string;
  x: number; y: number; size: number; scale: number; rotation: number; opacity: number;
  palette: PaletteSlot; color: string; text: string; weight: number; opticalSize: number;
  image: string;
};
export type ArtworkDocument = {
  version: 1; width: number; height: number; transparent: boolean;
  mode: 'tile' | 'poster'; repeat: 'grid' | 'half-drop';
  palette: Record<PaletteSlot, string>; objects: ArtworkObject[];
};
export function studioRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Invalid studio document. Your design is unchanged.');
  return value as Record<string, unknown>;
}
export function studioFields(value: Record<string, unknown>, keys: string[]) {
  if (Object.keys(value).some(key => !keys.includes(key))) throw Error('Unsupported studio field. Your design is unchanged.');
}
export function studioNumber(value: unknown, min: number, max: number) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw Error(`Studio value must be between ${min} and ${max}.`);
  return value;
}
export function studioDimension(value: unknown) {
  const dimension = studioNumber(value, 192, 1600);
  if (!Number.isInteger(dimension)) throw Error('Canvas dimensions must use whole pixels.');
  return dimension;
}
export function studioText(value: unknown, max = 300) {
  if (typeof value !== 'string' || value.length > max) throw Error(`Studio text supports up to ${max} characters.`);
  return value;
}
export function studioChoice<T extends string>(value: unknown, choices: readonly T[]): T {
  if (!choices.includes(value as T)) throw Error('Unsupported studio option.');
  return value as T;
}
export function studioColor(value: unknown) {
  const color = studioText(value, 7);
  if (!/^#[a-f\d]{6}$/i.test(color)) throw Error('Use a six-digit hex color.');
  return color;
}
export function studioImage(value: unknown) {
  const image = studioText(value, ARTWORK_LIMITS.imageBytes);
  if (!image) return image;
  const match = /^data:image\/(png|jpeg|webp);base64,([A-Za-z0-9+/]+=*)$/.exec(image);
  if (!match) throw Error('Upload an embedded PNG, JPEG, or WebP image.');
  let bytes: string;
  try { bytes = atob(match[2]); } catch { throw Error('Malformed image data.'); }
  const valid = match[1] === 'png' ? bytes.startsWith('\x89PNG\r\n\x1a\n') : match[1] === 'jpeg' ? bytes.startsWith('\xff\xd8\xff') : bytes.startsWith('RIFF') && bytes.slice(8, 12) === 'WEBP';
  if (!valid || btoa(bytes) !== match[2]) throw Error('Image data does not match its file type.');
  return image;
}
export function studioLatin(value: unknown) {
  const text = studioText(value);
  if (/[^\x20-\xff\n\r\t\u2000-\u206f\u2190-\u2193\u21a9]/.test(text)) throw Error('This bundled font edition supports Latin text and punctuation. Your previous text is kept.');
  return text;
}
export function newArtworkObject(kind: ArtworkKind = 'bloom'): ArtworkObject {
  return { id: `artwork-${crypto.randomUUID()}`, kind, name: kind === 'text' ? 'Your words' : kind, group: '', x: 50, y: 50, size: kind === 'text' ? 14 : 32, scale: 1, rotation: 0, opacity: 1, palette: 'ink', color: '', text: 'Good things\ntake shape.', weight: 650, opticalSize: 24, image: '' };
}
export function createArtwork(): ArtworkDocument {
  return { version: 1, width: 720, height: 720, transparent: false, mode: 'tile', repeat: 'grid', palette: { paper: '#f0eadc', ink: '#29382a', accent: '#c66a42' }, objects: [
    { ...newArtworkObject('bloom'), id: 'artwork-seed-bloom', x: 25, y: 25, size: 42 },
    { ...newArtworkObject('leaf'), id: 'artwork-seed-leaf', x: 75, y: 75, size: 40, rotation: 40, palette: 'accent' },
    { ...newArtworkObject('orbit'), id: 'artwork-seed-orbit', x: 78, y: 22, size: 17, palette: 'accent' },
  ] };
}
export function validateArtworkObject(input: unknown, prefix = 'artwork'): ArtworkObject {
  const o = studioRecord(input);
  studioFields(o, ['id', 'kind', 'name', 'group', 'x', 'y', 'size', 'scale', 'rotation', 'opacity', 'palette', 'color', 'text', 'weight', 'opticalSize', 'image']);
  const id = studioText(o.id, 80), group = studioText(o.group, 80);
  if (!new RegExp(`^${prefix}-[a-z0-9-]+$`).test(id) || (group && !/^group-[a-z0-9-]+$/.test(group))) throw Error('Invalid studio object identity.');
  const kind = studioChoice(o.kind, artworkKinds), image = studioImage(o.image);
  if (image && kind !== 'image') throw Error('Only image objects can contain uploads.');
  return { id, group, kind, name: studioText(o.name, 60), x: studioNumber(o.x, 0, 100), y: studioNumber(o.y, 0, 100), size: studioNumber(o.size, 1, 100), scale: studioNumber(o.scale, 0.1, 3), rotation: studioNumber(o.rotation, -180, 180), opacity: studioNumber(o.opacity, 0, 1), palette: studioChoice(o.palette, ['ink', 'accent', 'paper']), color: o.color === '' ? '' : studioColor(o.color), text: studioLatin(o.text), weight: studioNumber(o.weight, 100, 1000), opticalSize: studioNumber(o.opticalSize, 9, 40), image };
}
export function validateArtwork(input: unknown): ArtworkDocument {
  const a = studioRecord(input);
  studioFields(a, ['version', 'width', 'height', 'transparent', 'mode', 'repeat', 'palette', 'objects']);
  if (a.version !== ARTWORK_VERSION || typeof a.transparent !== 'boolean') throw Error('Unsupported artwork version or background setting.');
  if (!Array.isArray(a.objects) || a.objects.length > ARTWORK_LIMITS.objects) throw Error('Artwork supports up to 48 objects.');
  const palette = studioRecord(a.palette); studioFields(palette, ['ink', 'accent', 'paper']);
  const objects = a.objects.map(o => validateArtworkObject(o));
  if (new Set(objects.map(o => o.id)).size !== objects.length) throw Error('Duplicate artwork identity.');
  return { version: 1, width: studioDimension(a.width), height: studioDimension(a.height), transparent: a.transparent, mode: studioChoice(a.mode, ['tile', 'poster']), repeat: studioChoice(a.repeat, ['grid', 'half-drop']), palette: { ink: studioColor(palette.ink), accent: studioColor(palette.accent), paper: studioColor(palette.paper) }, objects };
}
export function artworkColor(a: ArtworkDocument, o: ArtworkObject) { return o.color || a.palette[o.palette]; }
export function duplicateArtwork(a: ArtworkDocument, ids: string[]): ArtworkDocument {
  const chosen = a.objects.filter(o => ids.includes(o.id));
  if (a.objects.length + chosen.length > ARTWORK_LIMITS.objects) throw Error('Artwork supports up to 48 objects.');
  const groups = new Map<string, string>();
  const copies = chosen.map(o => {
    if (o.group && !groups.has(o.group)) groups.set(o.group, `group-${crypto.randomUUID()}`);
    return { ...o, id: `artwork-${crypto.randomUUID()}`, name: `${o.name.slice(0, 50)} copy`, group: o.group ? groups.get(o.group)! : '', x: (o.x + 5) % 100, y: (o.y + 5) % 100 };
  });
  return validateArtwork({ ...a, objects: [...a.objects, ...copies] });
}
export function artworkPoster(style: 'editorial' | 'botanical' | 'orbit'): ArtworkDocument {
  const a = createArtwork(); a.mode = 'poster'; a.width = 720; a.height = 960;
  a.objects = [
    { ...newArtworkObject(style === 'botanical' ? 'leaf' : style === 'orbit' ? 'orbit' : 'bloom'), x: 52, y: 43, size: 80, palette: 'accent', rotation: style === 'botanical' ? -30 : 0 },
    { ...newArtworkObject('text'), x: 50, y: 18, text: style === 'botanical' ? 'A slower\nkind of living.' : style === 'orbit' ? 'A different\nperspective.' : 'Good things\ntake shape.', size: 11, weight: 750 },
    { ...newArtworkObject('text'), x: 50, y: 86, text: 'EIDOS WORKS / ORIGINAL STUDY', size: 2.8, weight: 500 },
  ];
  return validateArtwork(a);
}
