import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { verifyPrivateTemplateArchives } from '../scripts/check-template-archives.mjs';

test('paid deployment refuses missing or changed archives and duplicate inventory paths', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'template-build-'));
  const bytes = Buffer.from('fixture private archive bytes');
  const archives = [];
  try {
    for (const slug of ['switchboard', 'tideglass', 'matter', 'nightjar']) for (const edition of ['developer', 'wordpress']) {
      const key = `templates/${slug}/1.0.0/${edition}.zip`;
      await mkdir(path.dirname(path.join(root, key)), { recursive: true });
      await writeFile(path.join(root, key), bytes);
      archives.push({ archiveKey: key, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
    }
    const manifest = { schemaVersion: 1, archives };
    assert.equal(await verifyPrivateTemplateArchives(root, manifest), 8);
    await assert.rejects(verifyPrivateTemplateArchives(root, { ...manifest, archives: [...archives, archives[0]] }), /Invalid private/);
    await writeFile(path.join(root, archives[0].archiveKey), 'changed');
    await assert.rejects(verifyPrivateTemplateArchives(root, manifest), /differs from frozen/);
    await rm(path.join(root, archives[0].archiveKey));
    await assert.rejects(verifyPrivateTemplateArchives(root, manifest), { code: 'ENOENT' });
    const alternate = path.join(root, 'alternate');
    await writeFile(alternate, bytes);
    await symlink(alternate, path.join(root, archives[0].archiveKey));
    await assert.rejects(verifyPrivateTemplateArchives(root, manifest), /Unsafe private/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
