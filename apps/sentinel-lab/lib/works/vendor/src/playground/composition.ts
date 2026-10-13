import { sectionKind, validateProject, type Project } from './model';
import { defaultComposition, movePart, validateComposition, type Composition, type CompositionPart, type ImagePlacement } from './compositionSchema';
export type CompositionOperation =
  | { type: 'placement'; sectionId: string; placement: ImagePlacement; mobile?: boolean }
  | { type: 'move'; sectionId: string; part: CompositionPart; before: CompositionPart | null; mobile?: boolean }
  | { type: 'settings'; sectionId: string; patch: Partial<Composition> };
export function enableComposition(project: Project): Project {
  if (project.schemaVersion >= 3) return project;
  return validateProject({ ...project, schemaVersion: 3, sections: project.sections.map(section => ({
    ...section, type: sectionKind(section),
    ...(sectionKind(section) === 'hero' ? { composition: defaultComposition(section.layout) } : {}),
  })) });
}
/** Applies layout intent through the same validator as import, storage and cloud save. */
export function applyCompositionOperation(project: Project, operation: CompositionOperation): Project {
  if (![3,4].includes(project.schemaVersion)) throw Error('Enable spatial composition before moving elements.');
  const section = project.sections.find(value => value.id === operation.sectionId && sectionKind(value) === 'hero');
  if (!section?.composition) throw Error('This destination is not an editable hero.');
  let c = section.composition;
  if (operation.type === 'placement') {
    if (operation.mobile && (operation.placement === 'left' || operation.placement === 'right'))
      throw Error('Use a stacked image position on mobile.');
    c = { ...c, ...(operation.mobile ? { mobilePlacement: operation.placement as Composition['mobilePlacement'] } : { placement: operation.placement }) };
  } else if (operation.type === 'move') {
    c = { ...c, order: movePart(c.order, operation.part, operation.before), ...(operation.part === 'image' ? (operation.mobile ? { mobilePlacement: 'inline' as const } : { placement: 'inline' as const }) : {}) };
  } else if (operation.type === 'settings') {
    c = { ...c, ...operation.patch };
  } else throw Error('Unknown composition action.');
  const composition = validateComposition(c);
  return validateProject({ ...project, sections: project.sections.map(value => value.id === section.id ? { ...value, composition } : value) });
}
/** Stable IDs, not screen coordinates. Header/footer never become movable. */
export function moveSectionBefore(project: Project, id: string, beforeId: string): Project {
  const index = project.sections.findIndex(s => s.id === id);
  const before = project.sections.findIndex(s => s.id === beforeId);
  if (index <= 0 || index >= project.sections.length - 1 || before <= 0 || before > project.sections.length - 1)
    throw Error('Header and footer stay in place.');
  if (id === beforeId) return project;
  const sections = project.sections.filter(s => s.id !== id);
  sections.splice(sections.findIndex(s => s.id === beforeId), 0, project.sections[index]);
  return validateProject({ ...project, sections });
}
