import {applyBlockOperation} from './authoring';
import type { Card, Project, Section } from './model';
import type { SaveTarget } from './workspace';

const projectKeys = new Set(['schemaVersion', 'rendererVersion', 'template', 'name', 'tokens', 'glass', 'sections']);
const sectionKeys = new Set(['id', 'visible', 'title', 'description', 'cta', 'href', 'layout', 'image', 'alt']);
/** Do not remove until the paired hosted ownership/reopen tests pass. */
export function supportsProductionCloud(project: Project): boolean {
  return project.schemaVersion === 1
    && Object.keys(project).every(key => projectKeys.has(key))
    && project.sections.every(section => Object.keys(section).every(key => sectionKeys.has(key)));
}
/** Empty owner means UNREPORTED by the legacy API, never an authenticated identity.
 * Authorization remains server-side. New-protocol code must not trust this marker.
 */
export function legacySaveTarget(id: string, head: string, name: string, saved: string): SaveTarget {
  return { id, head, name, saved, owner: '' };
}
export function removeLocalImage(project: Project, id: string): Project {
  if(id.startsWith('node-'))return applyBlockOperation(project,{type:'patch',id,patch:{image:'',media:undefined}});
  const clear = <T extends Card | Section>(item: T): T => {
    const next = { ...item, image: '' };
    delete next.media;
    return next;
  };
  return { ...project, sections: project.sections.map(section => section.id === id ? clear(section) : {
    ...section, ...(section.cards ? { cards: section.cards.map(card => card.id === id ? clear(card) : card) } : {}),
  }) };
}
export const EDITOR_RELEASE = 'local-spatial-2026-09-10';
export const CLOUD_FORMAT_NOTICE = 'This design uses the new editor format. Cloud saving for this format is not enabled yet. Local autosave stays on; download project JSON or a page ZIP to keep a portable backup. Existing account projects are unchanged.';
