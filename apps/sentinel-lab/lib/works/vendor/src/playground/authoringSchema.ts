import { validateMedia, type MediaSettings } from './media';
export const BLOCK_LIMITS = { nodes: 96, sectionNodes: 48, depth: 3, documentBytes: 16000000 } as const;
export const nodeTypes = ['heading', 'text', 'image', 'button', 'stack', 'row', 'columns'] as const;
export type NodeType = typeof nodeTypes[number];
export type MobileLayout = {
    width?: number;
    height?: number;
    zoom?: number;
    gap?: number;
    align?: 'left' | 'center' | 'right';
    layout?: 'stack' | 'row';
};
export type BlockNode = {
    id: string;
    type: NodeType;
    name: string;
    text: string;
    href: string;
    image: string;
    alt: string;
    width: number;
    height: number;
    gap: number;
    align: 'left' | 'center' | 'right';
    split: number;
    lockAspect: boolean;
    media?: MediaSettings;
    mobile?: MobileLayout;
    children?: BlockNode[];
};
export type Authoring = {
    version: 1;
    root: BlockNode;
};
export const isContainer = (n: BlockNode) => ['stack', 'row', 'columns'].includes(n.type);
export const flattenNodes = (root: BlockNode): BlockNode[] => [root, ...(root.children || []).flatMap(flattenNodes)];
export function newNode(type: NodeType): BlockNode {
    return { id: 'node-' + crypto.randomUUID(), type, name: type === 'text' ? 'Text' : type.charAt(0).toUpperCase() + type.slice(1), text: type === 'heading' ? 'Your heading' : type === 'text' ? 'Write your own content.' : type === 'button' ? 'Learn more' : '', href: type === 'button' ? '#' : '', image: '', alt: '', width: 100, height: 320, gap: 24, align: 'left', split: 50, lockAspect: true, ...(['stack', 'row', 'columns'].includes(type) ? { children: [] } : {}) };
}
function object(v: unknown): Record<string, unknown> { if (!v || typeof v !== 'object' || Array.isArray(v))
    throw Error('Invalid responsive block.'); return v as Record<string, unknown>; }
function text(v: unknown, max: number) { if (typeof v !== 'string' || v.length > max)
    throw Error('Invalid or oversized block text.'); return v; }
function number(v: unknown, min: number, max: number) { if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max)
    throw Error('Block geometry is outside its supported range.'); return v; }
function choice<T extends string>(v: unknown, values: readonly T[]): T { if (typeof v !== 'string' || !values.includes(v as T))
    throw Error('Unsupported block option.'); return v as T; }
function keys(v: Record<string, unknown>, allowed: string[]) { if (Object.keys(v).some(k => !allowed.includes(k)))
    throw Error('Unsupported block field.'); }
export function validateBlockImage(v: unknown): string {
    const image = text(v, 4500000);
    if (!image)
        return '';
    const match = /^data:image\/(png|jpeg|webp);base64,([A-Za-z0-9+/]+=*)$/.exec(image);
    if (!match)
        throw Error('Images must be embedded PNG, JPEG or WebP.');
    let bytes: string;
    try {
        bytes = atob(match[2]);
    }
    catch {
        throw Error('Malformed image bytes.');
    }
    if (btoa(bytes) !== match[2] || !(match[1] === 'png' ? bytes.startsWith('\x89PNG\r\n\x1a\n') : match[1] === 'jpeg' ? bytes.startsWith('\xff\xd8\xff') : bytes.startsWith('RIFF') && bytes.slice(8, 12) === 'WEBP'))
        throw Error('Image signature does not match its type.');
    return image;
}
export function validateAuthoring(input: unknown): Authoring {
    const a = object(input);
    keys(a, ['version', 'root']);
    if (a.version !== 1)
        throw Error('Unsupported authoring version.');
    const seen = new Set<string>();
    let count = 0;
    function node(input: unknown, depth: number): BlockNode {
        if (depth > BLOCK_LIMITS.depth || ++count > BLOCK_LIMITS.sectionNodes)
            throw Error('Responsive block nesting/count limit exceeded.');
        const n = object(input);
        keys(n, ['id', 'type', 'name', 'text', 'href', 'image', 'alt', 'width', 'height', 'gap', 'align', 'split', 'lockAspect', 'media', 'mobile', 'children']);
        const id = text(n.id, 80);
        if (!/^node-[a-z0-9-]+$/.test(id) || seen.has(id))
            throw Error('Invalid or duplicate block identity.');
        seen.add(id);
        const type = choice(n.type, nodeTypes), container = ['stack', 'row', 'columns'].includes(type);
        if (container ? !Array.isArray(n.children) : n.children !== undefined)
            throw Error('Invalid block parent/child rule.');
        const href = text(n.href, 1000);
        if (href && !/^(#[\w-]*|mailto:[^\s<>"']+|tel:\+?[\d ()-]+|https?:\/\/[^\s<>"']+)$/i.test(href))
            throw Error('Use a safe link destination.');
        if (typeof n.lockAspect !== 'boolean')
            throw Error('Invalid aspect lock.');
        const image = validateBlockImage(n.image);
        if (type !== 'image' && (image || n.media !== undefined))
            throw Error('Only image blocks contain image media.');
        let mobile: MobileLayout | undefined;
        if (n.mobile !== undefined) {
            const m = object(n.mobile);
            keys(m, ['width', 'height', 'gap', 'align', 'layout', 'zoom']);
            mobile = { ...(m.zoom===undefined?{}:{zoom:number(m.zoom,1,3)}), ...(m.width === undefined ? {} : { width: number(m.width, 10, 100) }), ...(m.height === undefined ? {} : { height: number(m.height, 64, 960) }), ...(m.gap === undefined ? {} : { gap: number(m.gap, 0, 96) }), ...(m.align === undefined ? {} : { align: choice(m.align, ['left', 'center', 'right'] as const) }), ...(m.layout === undefined ? {} : { layout: choice(m.layout, ['stack', 'row'] as const) }) };
            if(type!=='image' && m.zoom!==undefined)throw Error('Only images have photo zoom.');
            if (!container && m.layout !== undefined)
                throw Error('Only containers have mobile layout.');
        }
        return { id, type, name: text(n.name, 60), text: text(n.text, type === 'heading' ? 240 : type === 'button' ? 100 : 8000), href, image, alt: text(n.alt, 300), width: number(n.width, 10, 100), height: number(n.height, 64, 960), gap: number(n.gap, 0, 96), align: choice(n.align, ['left', 'center', 'right'] as const), split: number(n.split, 20, 80), lockAspect: n.lockAspect, ...(n.media === undefined ? {} : { media: validateMedia(n.media) }), ...(mobile ? { mobile } : {}), ...(container ? { children: (n.children as unknown[]).map(v => node(v, depth + 1)) } : {}) };
    }
    const root = node(a.root, 0);
    if (!isContainer(root))
        throw Error('An authoring root must be a container.');
    return { version: 1, root };
}
