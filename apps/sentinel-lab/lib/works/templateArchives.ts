import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import type { PlatformEnv } from './vendor/functions/_shared/platform/core';

/** Private server bundle. Never place buyer ZIPs in Next public/ or a demo folder. */
export const templateArchives: NonNullable<PlatformEnv['EIDOS_TEMPLATE_ARCHIVES']> = {
  async get(key) {
    if (!/^templates\/(switchboard|tideglass|matter|nightjar)\/\d+\.\d+\.\d+\/(developer|wordpress)\.zip$/.test(key)) return null;
    try {
      const root = path.resolve(process.cwd(), 'private');
      const target = path.resolve(root, key);
      if (!target.startsWith(root + path.sep)) return null;
      const bytes = new Uint8Array(await readFile(target));
      return { bytes, sha256: createHash('sha256').update(bytes).digest('hex') };
    } catch { return null; }
  },
};
