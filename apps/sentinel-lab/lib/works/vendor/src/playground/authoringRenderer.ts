import { defaultMedia } from './media';
import type { Section } from './model';
import { flattenNodes, isContainer, type BlockNode } from './authoringSchema';
const esc = (s: string) => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
export function authoringMarkup(s: Section): string {
    let headings = 0;
    function node(n: BlockNode): string {
        const attrs = `id="${n.id}" class="pg-node pg-node-${n.type}" data-node="${n.id}" data-node-type="${n.type}" data-node-name="${esc(n.name)}" data-node-lock="${n.lockAspect}" data-node-mobile="${esc(JSON.stringify(n.mobile || {}))}" data-node-media="${esc(JSON.stringify(n.media || defaultMedia()))}"`;
        if (isContainer(n))
            return `<div ${attrs} data-node-container="true" data-node-empty="${!n.children!.length}">${n.children!.map(node).join('')}</div>`;
        if (n.type === 'image')
            return `<figure ${attrs}>${n.image ? `<div class="pg-block-frame"><img src="${esc(n.image)}" alt="${esc(n.alt)}" draggable="false"><i aria-hidden="true"></i></div>` : '<div class="pg-block-frame pg-block-empty" aria-label="Empty image frame"></div>'}</figure>`;
        if (n.type === 'heading') {
            const tag = s.type === 'hero' && headings++ === 0 ? 'h1' : 'h2';
            return `<div ${attrs}><${tag} data-node-text="true">${esc(n.text)}</${tag}></div>`;
        }
        if (n.type === 'button')
            return `<div ${attrs}><a class="pg-button" data-node-text="true" href="${esc(n.href || '#')}">${esc(n.text)}</a></div>`;
        return `<div ${attrs}><p data-node-text="true">${esc(n.text)}</p></div>`;
    }
    return `<section id="${s.id}" class="pg-authoring-section" data-section="${s.id}" data-authoring="true">${node(s.authoring!.root)}</section>`;
}
export function authoringStyles(sections: Section[]): string {
    const nodes = sections.flatMap(s => s.authoring ? flattenNodes(s.authoring.root) : []);
    if (!nodes.length)
        return '';
    const css = `.pg-authoring-section{padding:var(--space) clamp(18px,3vw,42px);min-width:0}.pg-node{position:relative;min-width:0;max-width:100%;margin:0;box-sizing:border-box;overflow-wrap:anywhere}.pg-node p,.pg-node h1,.pg-node h2{margin:0;max-width:none;white-space:pre-wrap}.pg-node-stack,.pg-node-row,.pg-node-columns{display:flex}.pg-node-stack{flex-direction:column}.pg-node[data-node-empty=true]{min-height:64px}.pg-node-row,.pg-node-columns{flex-direction:row;align-items:center}.pg-node-row>.pg-node,.pg-node-columns>.pg-node{flex:1 1 0}.pg-block-frame{height:100%;width:100%;position:relative;overflow:hidden;border-radius:var(--radius)}.pg-block-frame img{height:100%;width:100%;display:block}.pg-block-frame i{position:absolute;inset:0;pointer-events:none}.pg-block-empty{background:repeating-linear-gradient(135deg,#8882 0 12px,#8883 12px 24px);min-height:64px}.pg-node-image{flex-shrink:0}@media(max-width:640px){.pg-node-row,.pg-node-columns{flex-direction:column;align-items:stretch}.pg-node-row>.pg-node,.pg-node-columns>.pg-node{flex:0 0 auto!important;width:100%}.pg-authoring-section{padding:32px 18px}.pg-node h1{font-size:clamp(30px,10vw,48px)}}`;
    return css + nodes.map(n => {
        const id = '#' + n.id, m = n.type==='image' ? n.media || defaultMedia() : undefined, align = n.align === 'left' ? 'flex-start' : n.align === 'right' ? 'flex-end' : 'center';
        let rules = `${id}{width:${n.width}%;text-align:${n.align};gap:${n.gap}px;align-self:${align};${n.type === 'image' ? `height:${n.height}px;` : ''}}`;
        if (n.type === 'columns' && n.children?.length === 2)
            rules += `${id}>:first-child{flex:${n.split} 1 0}${id}>:last-child{flex:${100 - n.split} 1 0}`;
        if (m)
            rules += `${id} img{object-fit:${m.fit};object-position:${m.x}% ${m.y}%;transform:scale(${m.zoom})}${id} i{background:${m.tint};opacity:${m.opacity}}`;
        else
            rules += `${id} img{object-fit:cover}`;
        const mobile = n.mobile || {};
        rules += `@media(max-width:640px){${id}{width:${mobile.width ?? 100}%;gap:${mobile.gap ?? n.gap}px;text-align:${mobile.align ?? n.align};align-self:${mobile.align === 'right' ? 'flex-end' : mobile.align === 'center' ? 'center' : mobile.align === 'left' ? 'flex-start' : align};${n.type === 'image' ? `height:${mobile.height ?? Math.min(n.height, 360)}px;` : ''}${mobile.layout ? `flex-direction:${mobile.layout === 'row' ? 'row' : 'column'};` : ''}}${m ? `${id} img{object-position:${m.mobileX ?? m.x}% ${m.mobileY ?? m.y}%;transform:scale(${mobile.zoom ?? m.zoom})}` : ''}}`;
        if(mobile.layout==='row')rules+=`@media(max-width:640px){${id}>.pg-node{flex:1 1 0!important;width:auto}}`;
        return rules;
    }).join('');
}
