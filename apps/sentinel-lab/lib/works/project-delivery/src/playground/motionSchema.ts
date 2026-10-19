import { newArtworkObject, studioDimension, studioFields, studioNumber, studioRecord, studioChoice, studioColor, validateArtworkObject, type ArtworkObject } from './artworkSchema';
export type MotionFrame = { time: number; x: number; y: number; scale: number; rotation: number; opacity: number };
export type MotionObject = ArtworkObject & { keyframes: MotionFrame[] };
export type MotionDocument = { version: 1; width: number; height: number; background: string; duration: number; speed: number; loop: boolean; objects: MotionObject[] };
export const MOTION_LIMITS = { objects: 16, keyframes: 32, duration: 12 } as const;
export function motionFrame(object: Pick<ArtworkObject, 'x' | 'y' | 'scale' | 'rotation' | 'opacity'>, time: number): MotionFrame { return { time, x: object.x, y: object.y, scale: object.scale, rotation: object.rotation, opacity: object.opacity }; }
export function newMotionObject(kind: ArtworkObject['kind'] = 'bloom', duration = 4): MotionObject {
  const object = { ...newArtworkObject(kind), id: `motion-${crypto.randomUUID()}`, size: kind === 'text' ? 12 : 24, x: 50, y: 50 };
  return { ...object, keyframes: [motionFrame(object, 0), motionFrame(object, duration)] };
}
export function createMotion(): MotionDocument {
  const shape = { ...newMotionObject('bloom'), id: 'motion-seed-bloom', palette: 'accent' as const, color: '#c66a42' };
  shape.keyframes = [{ time: 0, x: 25, y: 50, scale: .6, rotation: -45, opacity: 1 }, { time: 2, x: 75, y: 50, scale: 1.2, rotation: 45, opacity: 1 }, { time: 4, x: 25, y: 50, scale: .6, rotation: -45, opacity: 1 }];
  const text = { ...newMotionObject('text'), id: 'motion-seed-text', text: 'Good things take shape.', size: 7, color: '#29382a', keyframes: [{ time: 0, x: 50, y: 82, scale: 1, rotation: 0, opacity: 1 }, { time: 4, x: 50, y: 82, scale: 1, rotation: 0, opacity: 1 }] };
  return { version: 1, width: 960, height: 640, background: '#f0eadc', duration: 4, speed: 1, loop: true, objects: [shape, text] };
}
export function validateMotion(input: unknown): MotionDocument {
  const m = studioRecord(input); studioFields(m, ['version', 'width', 'height', 'background', 'duration', 'speed', 'loop', 'objects']);
  if (m.version !== 1 || typeof m.loop !== 'boolean') throw Error('Unsupported motion version or playback option.');
  const duration = studioNumber(m.duration, 1, 12);
  if (!Array.isArray(m.objects) || m.objects.length > MOTION_LIMITS.objects) throw Error('Motion supports up to 16 objects.');
  const objects = m.objects.map(value => {
    const o = studioRecord(value); studioFields(o, ['id', 'kind', 'name', 'group', 'x', 'y', 'size', 'scale', 'rotation', 'opacity', 'palette', 'color', 'text', 'weight', 'opticalSize', 'image', 'keyframes']);
    const { keyframes, ...object } = o, validated = validateArtworkObject(object, 'motion');
    if (!Array.isArray(keyframes) || !keyframes.length || keyframes.length > MOTION_LIMITS.keyframes) throw Error('An animated object needs 1 to 32 keyframes.');
    const frames = keyframes.map(value => { const f = studioRecord(value); studioFields(f, ['time', 'x', 'y', 'scale', 'rotation', 'opacity']); return { time: studioNumber(f.time, 0, duration), x: studioNumber(f.x, 0, 100), y: studioNumber(f.y, 0, 100), scale: studioNumber(f.scale, .1, 3), rotation: studioNumber(f.rotation, -180, 180), opacity: studioNumber(f.opacity, 0, 1) }; }).sort((a, b) => a.time - b.time);
    if (new Set(frames.map(f => f.time)).size !== frames.length) throw Error('Each object can have only one keyframe at a given time.');
    return { ...validated, keyframes: frames };
  });
  if (new Set(objects.map(o => o.id)).size !== objects.length) throw Error('Duplicate motion identity.');
  return { version: 1, width: studioDimension(m.width), height: studioDimension(m.height), background: studioColor(m.background), duration, speed: studioNumber(m.speed, .25, 2), loop: m.loop, objects };
}
export function motionPose(object: MotionObject, time: number): MotionFrame {
  const frames = object.keyframes;
  if (time <= frames[0].time) return { ...frames[0], time };
  if (time >= frames[frames.length - 1].time) return { ...frames[frames.length - 1], time };
  const end = frames.findIndex(f => f.time >= time), from = frames[end - 1], to = frames[end], fraction = (time - from.time) / (to.time - from.time);
  const interpolate = (key: 'x' | 'y' | 'scale' | 'rotation' | 'opacity') => from[key] + (to[key] - from[key]) * fraction;
  return { time, x: interpolate('x'), y: interpolate('y'), scale: interpolate('scale'), rotation: interpolate('rotation'), opacity: interpolate('opacity') };
}
export function putMotionFrame(input: MotionDocument, id: string, frame: MotionFrame): MotionDocument {
  return validateMotion({ ...input, objects: input.objects.map(o => o.id === id ? { ...o, keyframes: [...o.keyframes.filter(f => f.time !== frame.time), frame] } : o) });
}
export function resizeMotionDuration(input: MotionDocument, duration: number): MotionDocument {
  studioNumber(duration, 1, 12);
  return validateMotion({ ...input, duration, objects: input.objects.map(o => ({ ...o, keyframes: o.keyframes.map(f => ({ ...f, time: Number((f.time / input.duration * duration).toFixed(6)) })) })) });
}
export function duplicateMotion(input: MotionDocument, id: string): MotionDocument {
  const o = input.objects.find(o => o.id === id); if (!o) return input;
  if (input.objects.length >= MOTION_LIMITS.objects) throw Error('Motion supports up to 16 objects.');
  return validateMotion({ ...input, objects: [...input.objects, { ...o, id: `motion-${crypto.randomUUID()}`, name: `${o.name.slice(0, 50)} copy`, group: '', keyframes: o.keyframes.map(f => ({ ...f, x: Math.min(100, f.x + 5), y: Math.min(100, f.y + 5) })) }] });
}
export const motionProperty = (value: unknown) => studioChoice(value, ['x', 'y', 'scale', 'rotation', 'opacity'] as const);
