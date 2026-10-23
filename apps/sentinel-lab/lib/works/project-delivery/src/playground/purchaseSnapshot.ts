import pins from '../data/buyerArchivePins.json';
import { exportFiles, zipFiles, type ExportFile } from './export';
import { mediaSlots, validateProject, type Project } from './model';

export const PACKAGE_FORMAT = 'eidos-local-page-package-v1';
export const parentEditions = pins.map(pin => ({ ...pin,
  productId: pin.family, platform: 'developer' as const,
  license: { id: 'eidos-commercial-developer-1.0', path: `products/templates/${pin.family}/developer/LICENSE.txt`, grant: 'not-verified' as const },
}));
export type ParentEdition = (typeof parentEditions)[number];
export type FileDigest = { path: string; bytes: number; sha256: string };
export type ProjectSnapshot = {
  schemaVersion: 1; kind: 'eidos-local-project-snapshot'; documentId: string; revision: number; pageId: 'page-main';
  project: Project; projectHash: string; snapshotHash: string; output: typeof PACKAGE_FORMAT; parent: ParentEdition | null;
};
export type PackageManifest = {
  schemaVersion: 1; kind: 'eidos-local-page-package'; snapshotHash: string; projectHash: string;
  documentId: string; revision: number; pageId: 'page-main'; rendererVersion: string;
  output: typeof PACKAGE_FORMAT; platform: 'developer'; parent: ParentEdition | null;
  entitlement: 'not-verified'; checkout: 'disabled'; files: FileDigest[];
};
export type FrozenPackage = {
  schemaVersion: 1; snapshot: ProjectSnapshot; manifest: PackageManifest; files: ExportFile[];
  archive: Uint8Array; receipt: { kind: 'eidos-local-package-receipt'; snapshotHash: string; archiveSha256: string; bytes: number; purchase: 'not-performed' };
};
const encode = (value: string) => new TextEncoder().encode(value);
const decode = (value: Uint8Array) => new TextDecoder('utf-8', { fatal: true }).decode(value);
const hashPattern = /^[a-f0-9]{64}$/;
const fileName = /^(?:[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\/)*[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$/;
const maxBytes = 48_000_000;

/** Stable JSON for identity; object key order is irrelevant, array order is not. */
export function canonicalJson(input: unknown): string {
  if (input === null || typeof input === 'string' || typeof input === 'boolean') return JSON.stringify(input);
  if (typeof input === 'number' && Number.isFinite(input)) return JSON.stringify(input);
  if (Array.isArray(input)) return '[' + input.map(canonicalJson).join(',') + ']';
  if (input && typeof input === 'object' && Object.getPrototypeOf(input) === Object.prototype)
    return '{' + Object.keys(input).filter(key => (input as Record<string, unknown>)[key] !== undefined).sort().map(key => JSON.stringify(key) + ':' + canonicalJson((input as Record<string, unknown>)[key])).join(',') + '}';
  throw Error('Snapshot contains an unsupported value.');
}
export async function hashBytes(bytes: Uint8Array): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer))].map(byte => byte.toString(16).padStart(2, '0')).join('');
}
const hashJson = (value: unknown) => hashBytes(encode(canonicalJson(value)));
function record(input: unknown, keys: string[]): Record<string, unknown> {
  if (!input || typeof input !== 'object' || Array.isArray(input) || Object.keys(input).some(key => !keys.includes(key)) || keys.some(key => !(key in input))) throw Error('Unsupported or incomplete snapshot record.');
  return input as Record<string, unknown>;
}
function identity(documentId: unknown, revision: unknown) {
  if (typeof documentId !== 'string' || !/^[a-zA-Z0-9-]{1,100}$/.test(documentId) || !Number.isSafeInteger(revision) || Number(revision) < 0) throw Error('A stable project identity and revision are required.');
}
function parentReference(input: unknown): ParentEdition | null {
  if (input === null) return null;
  const match = parentEditions.find(parent => canonicalJson(parent) === canonicalJson(input));
  if (!match) throw Error('The parent edition reference does not match the retained registry.');
  return structuredClone(match);
}
function pageProject(input: unknown): Project {
  if (encode(JSON.stringify(input)).length > 16_000_000) throw Error('Frozen project exceeds the 16 MB source limit.');
  const value = validateProject(input);
  if (value.studios && value.studios.active !== 'page') throw Error('Frozen review currently supports single-page output. Keep specialized exports in their existing tools.');
  return value;
}

/** Container completeness checks precede the real browser decode in Finish. */
export function rasterBytes(data: string): Uint8Array {
  const match = /^data:image\/(png|jpeg|webp);base64,([A-Za-z0-9+/]+=*)$/.exec(data);
  if (!match || data.length > 4_500_000) throw Error('Only embedded PNG, JPEG, and WebP images can be frozen.');
  const binary = atob(match[2]);
  if (btoa(binary) !== match[2]) throw Error('Image encoding is invalid.');
  const bytes = Uint8Array.from(binary, ch => ch.charCodeAt(0));
  const text = (start: number, length: number) => String.fromCharCode(...bytes.subarray(start, start + length));
  const view = new DataView(bytes.buffer); let width = 0, height = 0;
  if (match[1] === 'png') {
    if (bytes.length < 57 || text(0, 8) !== '\x89PNG\r\n\x1a\n' || text(12, 4) !== 'IHDR' || view.getUint32(8) !== 13) throw Error('PNG image is incomplete.');
    width = view.getUint32(16); height = view.getUint32(20);
    let offset = 8, ended = false, hasData = false;
    while (offset + 12 <= bytes.length) {
      const size = view.getUint32(offset), type = text(offset + 4, 4);
      if (size > bytes.length - offset - 12) throw Error('PNG image is incomplete.');
      hasData ||= type === 'IDAT'; offset += size + 12;
      if (type === 'IEND') { ended = size === 0 && offset === bytes.length; break; }
    }
    if (!ended || !hasData) throw Error('PNG image is incomplete.');
  } else if (match[1] === 'jpeg') {
    if (bytes.length < 8 || view.getUint16(0) !== 0xffd8 || view.getUint16(bytes.length - 2) !== 0xffd9) throw Error('JPEG image is incomplete.');
    let offset = 2;
    while (offset + 4 <= bytes.length) {
      if (bytes[offset++] !== 0xff) throw Error('JPEG image is malformed.');
      while (bytes[offset] === 0xff) offset++;
      const marker = bytes[offset++]; if (marker === 0xda || marker === 0xd9) break;
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (offset + 2 > bytes.length) throw Error('JPEG image is incomplete.');
      const size = view.getUint16(offset);
      if (size < 2 || offset + size > bytes.length) throw Error('JPEG image is incomplete.');
      if ([0xc0, 0xc1, 0xc2].includes(marker) && size >= 8) { height = view.getUint16(offset + 3); width = view.getUint16(offset + 5); }
      offset += size;
    }
  } else {
    if (bytes.length < 30 || text(0, 4) !== 'RIFF' || text(8, 4) !== 'WEBP' || view.getUint32(4, true) + 8 !== bytes.length) throw Error('WebP image is incomplete.');
    const format = text(12, 4), size = view.getUint32(16, true);
    if (20 + size + (size % 2) > bytes.length) throw Error('WebP image is incomplete.');
    if (format === 'VP8X' && size === 10) { width = 1 + bytes[24] + bytes[25] * 256 + bytes[26] * 65536; height = 1 + bytes[27] + bytes[28] * 256 + bytes[29] * 65536; }
    else if (format === 'VP8 ' && size >= 10 && text(23, 3) === '\x9d\x01\x2a') { width = view.getUint16(26, true) & 16383; height = view.getUint16(28, true) & 16383; }
    else if (format === 'VP8L' && size >= 5 && bytes[20] === 0x2f) { const bits = view.getUint32(21, true); width = 1 + (bits & 16383); height = 1 + ((bits >>> 14) & 16383); }
  }
  if (!width || !height || width > 12000 || height > 12000 || width * height > 16_000_000) throw Error('Image dimensions are invalid or exceed the 16 megapixel limit.');
  return bytes;
}
export function snapshotImageSlots(project: Project): { image: string }[] {
  return [...mediaSlots(project).map(slot => ({ image: slot.image || '' })), ...(project.studios?.artwork?.objects || []), ...(project.studios?.motion?.objects || [])];
}
export async function decodeSnapshotImages(project: Project): Promise<void> {
  for (const slot of snapshotImageSlots(project)) if (slot.image) {
    rasterBytes(slot.image);
    const image = new Image(); image.src = slot.image;
    try { await image.decode(); } catch { throw Error('An image cannot be decoded. Re-upload it before freezing this page.'); }
    if (!image.naturalWidth || !image.naturalHeight || image.naturalWidth * image.naturalHeight > 16_000_000) throw Error('An image has invalid or oversized dimensions.');
  }
}
export async function createProjectSnapshot(input: Project, documentId: string, revision: number, parentEditionId = ''): Promise<ProjectSnapshot> {
  identity(documentId, revision);
  const project = structuredClone(pageProject(input));
  for (const slot of snapshotImageSlots(project)) if (slot.image) rasterBytes(slot.image);
  const parent = parentEditionId ? parentEditions.find(value => value.editionId === parentEditionId) : null;
  if (parentEditionId && !parent) throw Error('Choose an existing developer edition reference.');
  const payload: Omit<ProjectSnapshot, 'snapshotHash'> = { schemaVersion: 1, kind: 'eidos-local-project-snapshot', documentId, revision, pageId: 'page-main', project, projectHash: await hashJson(project), output: PACKAGE_FORMAT, parent: parent ? structuredClone(parent) : null };
  return { ...payload, snapshotHash: await hashJson(payload) };
}
export async function validateProjectSnapshot(input: unknown): Promise<ProjectSnapshot> {
  const raw = record(input, ['schemaVersion', 'kind', 'documentId', 'revision', 'pageId', 'project', 'projectHash', 'snapshotHash', 'output', 'parent']);
  if (raw.schemaVersion !== 1 || raw.kind !== 'eidos-local-project-snapshot' || raw.pageId !== 'page-main' || raw.output !== PACKAGE_FORMAT || !hashPattern.test(String(raw.projectHash)) || !hashPattern.test(String(raw.snapshotHash))) throw Error('Unsupported frozen project snapshot.');
  identity(raw.documentId, raw.revision);
  const project = pageProject(raw.project), parent = parentReference(raw.parent);
  const checked = await createProjectSnapshot(project, raw.documentId as string, raw.revision as number, parent?.editionId);
  if (canonicalJson(checked) !== canonicalJson(raw)) throw Error('Frozen project content or identity hash does not match.');
  return checked;
}
/** Stored receipts retain old parent identities. New checkout still uses the
 * stricter current-registry validator above; this reader grants no rights. */
export async function validateStoredProjectSnapshot(input: unknown): Promise<ProjectSnapshot> {
  const raw = record(input, ['schemaVersion', 'kind', 'documentId', 'revision', 'pageId', 'project', 'projectHash', 'snapshotHash', 'output', 'parent']);
  if (raw.schemaVersion !== 1 || raw.kind !== 'eidos-local-project-snapshot' || raw.pageId !== 'page-main' || raw.output !== PACKAGE_FORMAT || !hashPattern.test(String(raw.projectHash)) || !hashPattern.test(String(raw.snapshotHash))) throw Error('This stored source needs another Playground version.');
  identity(raw.documentId, raw.revision);
  const project = pageProject(raw.project);
  if (canonicalJson(project) !== canonicalJson(raw.project)) throw Error('This source cannot be reopened without changing its saved content.');
  if (raw.parent !== null) {
    const parent = record(raw.parent, ['family', 'editionId', 'version', 'sha256', 'bytes', 'catalogVersion', 'status', 'productId', 'platform', 'license']);
    const license = record(parent.license, ['id', 'path', 'grant']);
    if (typeof parent.family !== 'string' || !/^[a-z][a-z0-9-]{0,50}$/.test(parent.family) || parent.productId !== parent.family || parent.platform !== 'developer' || typeof parent.editionId !== 'string' || !/^[a-z0-9-]{1,100}$/.test(parent.editionId) || typeof parent.version !== 'string' || parent.version.length > 40 || !hashPattern.test(String(parent.sha256)) || !Number.isSafeInteger(parent.bytes) || Number(parent.bytes) <= 0 || typeof parent.catalogVersion !== 'string' || parent.catalogVersion.length > 100 || typeof parent.status !== 'string' || parent.status.length > 300 || typeof license.id !== 'string' || !/^[a-z0-9.-]{1,100}$/.test(license.id) || license.path !== `products/templates/${parent.family}/developer/LICENSE.txt` || license.grant !== 'not-verified') throw Error('Stored parent provenance is malformed.');
  }
  const { snapshotHash, ...payload } = raw;
  if (await hashJson(project) !== raw.projectHash || await hashJson(payload) !== snapshotHash) throw Error('Stored snapshot identity or content hash does not match.');
  return structuredClone(raw) as ProjectSnapshot;
}
export async function digestFiles(files: ExportFile[]): Promise<FileDigest[]> {
  const seen = new Set<string>(); let total = 0;
  if (files.length < 1 || files.length > 200) throw Error('Invalid package file count.');
  return Promise.all(files.map(async file => {
    if (!fileName.test(file.name) || seen.has(file.name.toLowerCase()) || !(file.data instanceof Uint8Array) || (total += file.data.length) > maxBytes) throw Error('Unsafe, duplicate, or oversized package file.');
    seen.add(file.name.toLowerCase());
    return { path: file.name, bytes: file.data.length, sha256: await hashBytes(file.data) };
  }));
}
const provenance = (snapshot: ProjectSnapshot) => `# Local page package\n\nThis package contains the exact reviewed single-page export and editable project. It is not a paid template archive, order, receipt of payment, or verified commercial license. No hosting, domain, backend, form delivery, or CMS installation is included.\n\nDocument: ${snapshot.documentId}\nRevision: ${snapshot.revision}\nSnapshot SHA-256: ${snapshot.snapshotHash}\nOutput: ${PACKAGE_FORMAT} (standalone HTML/CSS/JavaScript)\n\n${snapshot.parent ? `Retained reference only: ${snapshot.parent.productId} / ${snapshot.parent.editionId} / ${snapshot.parent.version}. Original archive SHA-256: ${snapshot.parent.sha256}; bytes: ${snapshot.parent.bytes}. License reference: ${snapshot.parent.license.id}, ${snapshot.parent.license.path}. The original archive is not included or modified. Selecting this reference does not establish ownership or grant rights. Keep the original licensed source and its notices.\n\n` : 'No purchased parent edition has been asserted.\n\n'}Eidos starter illustrations and renderer retain their existing rights; user uploads require your own rights. This local review grants no additional license. Do not relabel it as a paid family package.\n\nExtract all files together and open index.html. Import project.json into Playground to resume editing; importing creates a working draft. The frozen package remains separate. SNAPSHOT.json identifies the reviewed document/revision; PACKAGE_MANIFEST.json hashes every other file, including the source and provenance. The separate download receipt hashes the complete ZIP (a manifest cannot hash itself). The frozen ZIP remains the same reviewed artifact before and after any purchase. Payment and entitlement are established separately by a verified order receipt, not by this ZIP. When an original edition is purchased, its licensed archive and notices are delivered separately and remain unmodified. Keep that original alongside this approved page package.\n`;

export async function buildFrozenPackage(input: ProjectSnapshot): Promise<FrozenPackage> {
  const snapshot = await validateProjectSnapshot(input);
  const files = [...exportFiles(snapshot.project),
    { name: 'SNAPSHOT.json', data: encode(JSON.stringify(snapshot, null, 2)) },
    { name: 'PROVENANCE.md', data: encode(provenance(snapshot)) },
  ];
  const manifest: PackageManifest = { schemaVersion: 1, kind: 'eidos-local-page-package', snapshotHash: snapshot.snapshotHash, projectHash: snapshot.projectHash, documentId: snapshot.documentId, revision: snapshot.revision, pageId: 'page-main', rendererVersion: snapshot.project.rendererVersion, output: PACKAGE_FORMAT, platform: 'developer', parent: snapshot.parent, entitlement: 'not-verified', checkout: 'disabled', files: await digestFiles(files) };
  files.push({ name: 'PACKAGE_MANIFEST.json', data: encode(JSON.stringify(manifest, null, 2)) });
  const archive = zipFiles(files);
  if (archive.length > maxBytes) throw Error('Frozen package exceeds the 48 MB local limit.');
  return { schemaVersion: 1, snapshot, manifest, files, archive, receipt: { kind: 'eidos-local-package-receipt', snapshotHash: snapshot.snapshotHash, archiveSha256: await hashBytes(archive), bytes: archive.length, purchase: 'not-performed' } };
}
/** Read old compiled bytes as stored: do not silently re-render with newer code. */
export async function verifyFrozenPackage(input: FrozenPackage): Promise<FrozenPackage> {
  record(input, ['schemaVersion', 'snapshot', 'manifest', 'files', 'archive', 'receipt']);
  if (input.schemaVersion !== 1 || !Array.isArray(input.files) || !(input.archive instanceof Uint8Array) || input.archive.length > maxBytes) throw Error('Unsupported frozen package.');
  const snapshot = await validateStoredProjectSnapshot(input.snapshot), manifestFile = input.files.find(file => file.name === 'PACKAGE_MANIFEST.json');
  if (!manifestFile) throw Error('Frozen package manifest is missing.');
  const digests = await digestFiles(input.files.filter(file => file.name !== 'PACKAGE_MANIFEST.json'));
  const expected: PackageManifest = { schemaVersion: 1, kind: 'eidos-local-page-package', snapshotHash: snapshot.snapshotHash, projectHash: snapshot.projectHash, documentId: snapshot.documentId, revision: snapshot.revision, pageId: 'page-main', rendererVersion: snapshot.project.rendererVersion, output: PACKAGE_FORMAT, platform: 'developer', parent: snapshot.parent, entitlement: 'not-verified', checkout: 'disabled', files: digests };
  if (canonicalJson(input.manifest) !== canonicalJson(expected) || canonicalJson(JSON.parse(decode(manifestFile.data))) !== canonicalJson(expected)) throw Error('Frozen package manifest or file hashes do not match.');
  await digestFiles(input.files);
  const source = input.files.find(file => file.name === 'project.json'), sourceSnapshot = input.files.find(file => file.name === 'SNAPSHOT.json');
  if (!source || !sourceSnapshot || canonicalJson(JSON.parse(decode(source.data))) !== canonicalJson(snapshot.project) || canonicalJson(JSON.parse(decode(sourceSnapshot.data))) !== canonicalJson(snapshot)) throw Error('Frozen editable source does not match its snapshot.');
  const receipt = { kind: 'eidos-local-package-receipt', snapshotHash: snapshot.snapshotHash, archiveSha256: await hashBytes(input.archive), bytes: input.archive.length, purchase: 'not-performed' };
  if (canonicalJson(input.receipt) !== canonicalJson(receipt) || await hashBytes(zipFiles(input.files)) !== receipt.archiveSha256) throw Error('Frozen archive bytes or receipt do not match.');
  return input;
}
export function isCurrentReview(snapshot: ProjectSnapshot, documentId: string, revision: number): boolean { return snapshot.documentId === documentId && snapshot.revision === revision; }

/** Parse only our deterministic ZIP STORE format; verification re-encodes it
 * byte-for-byte, checking the complete central directory and all CRC fields. */
export async function frozenPackageFromArchive(archive: Uint8Array): Promise<FrozenPackage> {
  if (archive.length < 22 || archive.length > maxBytes) throw Error('Stored archive size is unsupported.');
  const files: ExportFile[] = [], view = new DataView(archive.buffer, archive.byteOffset, archive.byteLength); let offset = 0;
  while (offset + 30 <= archive.length && view.getUint32(offset, true) === 0x04034b50) {
    const size = view.getUint32(offset + 18, true), raw = view.getUint32(offset + 22, true), nameLength = view.getUint16(offset + 26, true), extra = view.getUint16(offset + 28, true), start = offset + 30 + nameLength + extra;
    if (files.length >= 200 || view.getUint16(offset + 6, true) !== 0x800 || view.getUint16(offset + 8, true) !== 0 || raw !== size || !nameLength || extra || start + size > archive.length) throw Error('Stored archive format is unsupported. Keep the verified ZIP for manual use.');
    files.push({ name: decode(archive.slice(offset + 30, offset + 30 + nameLength)), data: archive.slice(start, start + size) }); offset = start + size;
  }
  await digestFiles(files);
  const snapshotFile = files.find(file => file.name === 'SNAPSHOT.json'), manifestFile = files.find(file => file.name === 'PACKAGE_MANIFEST.json');
  if (!snapshotFile || !manifestFile) throw Error('The stored archive has no supported editable snapshot.');
  const snapshot = await validateStoredProjectSnapshot(JSON.parse(decode(snapshotFile.data))), manifest = JSON.parse(decode(manifestFile.data)) as PackageManifest;
  return verifyFrozenPackage({ schemaVersion: 1, snapshot, manifest, files, archive, receipt: { kind: 'eidos-local-package-receipt', snapshotHash: snapshot.snapshotHash, archiveSha256: await hashBytes(archive), bytes: archive.length, purchase: 'not-performed' } });
}

/** Preview the packaged HTML/CSS/script bytes with local assets and no network. */
export function frozenPreviewDocument(frozen: FrozenPackage): string {
  const text = (name: string) => { const file = frozen.files.find(file => file.name === name); if (!file) throw Error(`Frozen package is missing ${name}.`); return decode(file.data); };
  let html = text('index.html').replace('<link rel="stylesheet" href="tokens.css">', () => `<style>${text('tokens.css')}</style>`).replace('<link rel="stylesheet" href="styles.css">', () => `<style>${text('styles.css')}</style>`).replace('<script src="script.js" defer></script>', () => `<script>${text('script.js')}</script>`);
  for (const file of frozen.files.filter(file => file.name.startsWith('assets/'))) {
    const type = file.name.endsWith('.jpg') ? 'jpeg' : file.name.endsWith('.png') ? 'png' : 'webp';
    let binary = ''; for (const byte of file.data) binary += String.fromCharCode(byte);
    html = html.replaceAll(`src="${file.name}"`, `src="data:image/${type};base64,${btoa(binary)}"`);
  }
  return html.replace('<head>', '<head><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'; form-action \'none\'; base-uri \'none\'">').replace('</body>', '<script>document.addEventListener("click",e=>{const a=e.target.closest("a");if(a&&!a.getAttribute("href").startsWith("#"))e.preventDefault()},true)</script></body>');
}
