import { HttpError, type PlatformEnv } from './core';

export const LOCAL_STUDIO_CLOUD_NOTICE = 'Studio projects are saved on this device. Export a project JSON backup; account saving for this format awaits paired hosted ownership and reopen acceptance.';

export function assertProjectSaveBoundary(env: PlatformEnv, input: Record<string, unknown>) {
  if (input.intent === 'resume') {
    if (env.EIDOS_PLAYGROUND_RESUME_ENABLED !== 'true') throw new HttpError(409, LOCAL_STUDIO_CLOUD_NOTICE);
    if (typeof input.expectedOwner !== 'string' || !input.expectedOwner) throw new HttpError(409, 'Refresh your account before saving this project.');
    return;
  }
  assertCloudStudioBoundary(input.document);
}

/** Reject new local-only documents before an older validator can discard fields. */
export function assertCloudStudioBoundary(document: unknown): void {
  if (!document || typeof document !== 'object' || Array.isArray(document)) return;
  const project = document as Record<string, unknown>;
  if ((typeof project.schemaVersion === 'number' && project.schemaVersion >= 5)
    || Object.prototype.hasOwnProperty.call(project, 'studios')) {
    throw new HttpError(409, LOCAL_STUDIO_CLOUD_NOTICE);
  }
}
