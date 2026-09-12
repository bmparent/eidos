import { validateProject, sectionKind, newBlock, type Project } from './model';
import { newNode, flattenNodes, isContainer, type BlockNode, type NodeType } from './authoringSchema';
export type BlockOperation = {
    type: 'section-step';
    id: string;
    direction: -1 | 1;
} | {
    type: 'card-step';
    id: string;
    direction: -1 | 1;
} | {
    type: 'add';
    parent: string;
    kind: NodeType;
} | {
    type: 'move';
    id: string;
    parent: string;
    before: string | null;
} | {
    type: 'duplicate';
    id: string;
} | {
    type: 'remove';
    id: string;
} | {
    type: 'patch';
    id: string;
    patch: Partial<BlockNode>;
} | {
    type: 'add-section';
};
export function enableBlocks(p: Project): Project {
    if (p.schemaVersion === 4)
        return p;
    return validateProject({ ...p, schemaVersion: 4, sections: p.sections.map(s => {
            if (sectionKind(s) !== 'hero')
                return { ...s, type: sectionKind(s) };
            const root = newNode('stack');
            root.name = 'Hero content';
            root.align = s.composition?.align || 'left';
            root.gap = s.composition?.gap ?? 24;
            const nodes: Record<string, BlockNode> = { title: { ...newNode('heading'), name: 'Heading', text: s.title }, description: { ...newNode('text'), name: 'Description', text: s.description }, cta: { ...newNode('button'), name: 'Button', text: s.cta, href: s.href }, image: { ...newNode('image'), name: 'Hero image', image: s.image, alt: s.alt, ...(s.media ? { media: s.media } : {}) } };
            root.children = (s.composition?.order || ['title', 'description', 'cta', 'image']).map(part => nodes[part]);
            // Legacy fields remain intact for provenance. Only explicit upgrade chooses responsive flow.
            return { ...s, type: sectionKind(s), authoring: { version: 1, root } };
        }) });
}
export function applyBlockOperation(p: Project, op: BlockOperation): Project {
    if (p.schemaVersion !== 4)
        throw Error('Enable responsive blocks first.');
    const next = structuredClone(p);
    if (op.type === 'section-step' || op.type === 'card-step') {
        if (op.direction !== -1 && op.direction !== 1) throw Error('Invalid move direction.');
        const list = op.type === 'section-step' ? next.sections : next.sections.find(s => s.cards?.some(c => c.id === op.id))?.cards;
        if (!list) throw Error('Card no longer exists.');
        const from = list.findIndex(n => n.id === op.id), to = from + op.direction;
        if (from < 0) throw Error('Item no longer exists.');
        const pinned = op.type === 'section-step';
        if (from < (pinned ? 1 : 0) || from >= list.length - (pinned ? 1 : 0) || to < (pinned ? 1 : 0) || to >= list.length - (pinned ? 1 : 0)) return p;
        [list[from], list[to]] = [list[to], list[from]];
        return validateProject(next);
    }
    const roots = next.sections.flatMap(s => s.authoring ? [s.authoring.root] : []);
    const all = roots.flatMap(flattenNodes);
    const find = (id: string) => { const node = all.find(n => n.id === id); if (!node)
        throw Error('Block no longer exists.'); return node; };
    const parentOf = (id: string) => all.find(n => n.children?.some(c => c.id === id));
    if (op.type === 'add-section') {
        const s = newBlock('content'), root = newNode('stack');
        s.title = 'Content section';
        root.name = 'Content';
        root.children = [newNode('text')];
        next.sections.splice(next.sections.length - 1, 0, { ...s, authoring: { version: 1, root } });
    }
    else if (op.type === 'add') {
        const parent = find(op.parent);
        if (!isContainer(parent))
            throw Error('Choose a container.');
        parent.children!.push(newNode(op.kind));
    }
    else if (op.type === 'patch') {
        const node = find(op.id);
        if (Object.keys(op.patch).some(k => ['id', 'type', 'children'].includes(k)))
            throw Error('Identity and structure need a dedicated command.');
        Object.assign(node, op.patch);
    }
    else {
        const node = find(op.id), parent = parentOf(op.id);
        if (!parent)
            throw Error('A section root stays attached to its section.');
        if (op.type === 'remove')
            parent.children = parent.children!.filter(n => n.id !== node.id);
        else if (op.type === 'duplicate') {
            const clone = structuredClone(node);
            for (const n of flattenNodes(clone))
                n.id = newNode(n.type).id;
            clone.name = (clone.name + ' copy').slice(0, 60);
            parent.children!.splice(parent.children!.indexOf(node) + 1, 0, clone);
        }
        else {
            const target = find(op.parent);
            if (!isContainer(target) || flattenNodes(node).some(n => n.id === target.id))
                throw Error('Cannot move a container into itself or its descendants.');
            if (op.before === node.id && target === parent)
                return p;
            if (op.before !== null && !target.children!.some(n => n.id === op.before))
                throw Error('Destination is not inside that container.');
            parent.children = parent.children!.filter(n => n.id !== node.id);
            const index = op.before === null ? target.children!.length : target.children!.findIndex(n => n.id === op.before);
            target.children!.splice(index, 0, node);
        }
    }
    const validated = validateProject(next);
    return JSON.stringify(validated) === JSON.stringify(p) ? p : validated;
}
