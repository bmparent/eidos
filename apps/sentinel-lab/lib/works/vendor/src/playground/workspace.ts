import type { Project } from './model';

export type SaveTarget = { id: string; head: string; owner: string; name: string; saved: string };
export type Frame = { project: Project; documentId: string; target: SaveTarget | null };
export type Workspace = { current: Frame; past: Frame[]; future: Frame[]; group: string; time: number };
export type Action =
  | { type: 'edit'; project: Project | ((p: Project) => Project); group?: string; time: number }
  | { type: 'replace'; project: Project; documentId: string; target?: SaveTarget | null }
  | { type: 'load'; project: Project; documentId: string }
  | { type: 'saved'; documentId: string; target: SaveTarget }
  | { type: 'undo' } | { type: 'redo' };
export function workspace(project: Project, documentId: string): Workspace {
  return { current: { project, documentId, target: null }, past: [], future: [], group: '', time: 0 };
}
export function workspaceReducer(s: Workspace, a: Action): Workspace {
  if (a.type === 'load') return workspace(a.project, a.documentId);
  if (a.type === 'undo') return s.past.length ? { current: s.past.at(-1)!, past: s.past.slice(0,-1), future: [s.current,...s.future], group: '', time: 0 } : s;
  if (a.type === 'redo') return s.future.length ? { current: s.future[0], past: [...s.past,s.current], future: s.future.slice(1), group: '', time: 0 } : s;
  if (a.type === 'saved') {
    // Update the head for this document's history, never for another document.
    const update = (f: Frame): Frame => f.documentId === a.documentId ? {...f,target:a.target} : f;
    return {...s,current:update(s.current),past:s.past.map(update),future:s.future.map(update)};
  }
  const current: Frame = a.type === 'replace'
    ? {project:a.project,documentId:a.documentId,target:a.target || null}
    : {...s.current,project:typeof a.project === 'function' ? a.project(s.current.project) : a.project};
  return {current,past:a.type === 'edit' && a.group && a.group === s.group && a.time-s.time < 900 ? s.past : [...s.past,s.current].slice(-30),future:[],group:a.type === 'edit' ? a.group || '' : '',time:a.type === 'edit' ? a.time : 0};
}
