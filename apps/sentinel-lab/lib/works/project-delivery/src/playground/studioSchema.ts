import { validateArtwork, type ArtworkDocument } from './artworkSchema';
import { validateMotion, type MotionDocument } from './motionSchema';

export const studioModes = ['page', 'effect', 'artwork', 'layout', 'motion'] as const;
export type StudioMode = typeof studioModes[number];
export type Studios = { version: 1; active: StudioMode; artwork?: ArtworkDocument; motion?: MotionDocument };
export function validateStudios(input: unknown): Studios {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw Error('Invalid studio document.');
  const v = input as Record<string, unknown>;
  if (v.version !== 1 || !studioModes.includes(v.active as StudioMode) || Object.keys(v).some(k => !['version','active','artwork','motion'].includes(k))) throw Error('Unsupported studio version or field. Your open design is unchanged.');
  return { version: 1, active: v.active as StudioMode, ...(v.artwork === undefined ? {} : {artwork: validateArtwork(v.artwork)}), ...(v.motion === undefined ? {} : {motion: validateMotion(v.motion)}) };
}
