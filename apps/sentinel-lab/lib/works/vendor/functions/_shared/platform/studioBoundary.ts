import { HttpError } from './core';

export const LOCAL_STUDIO_CLOUD_NOTICE = 'Studio projects are saved on this device. Export a project JSON backup; account saving for this format awaits paired hosted ownership and reopen acceptance.';

/** Reject new local-only documents before an older validator can discard fields. */
export function assertCloudStudioBoundary(document: unknown): void {
  if (!document || typeof document !== 'object' || Array.isArray(document)) return;
  const project = document as Record<string, unknown>;
  if ((typeof project.schemaVersion === 'number' && project.schemaVersion >= 5)
    || Object.prototype.hasOwnProperty.call(project, 'studios')) {
    throw new HttpError(409, LOCAL_STUDIO_CLOUD_NOTICE);
  }
}
