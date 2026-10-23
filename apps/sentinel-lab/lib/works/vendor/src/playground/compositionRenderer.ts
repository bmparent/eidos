import type { Section } from './model';
import { effectivePlacement, type Composition, type ImagePlacement } from './compositionSchema';
/** The DOM order is the reading order. The background remains an image with its original alt text. */
export function compositionMarkup(section: Section, html: { title: string; description: string; cta: string; image: string }): string {
  const c = section.composition!;
  const part = (name: keyof typeof html) => `<div class="pg-part pg-part-${name}" data-composition-part="${name}">${html[name]}</div>`;
  // An empty CTA/description stays editable, without an empty focus target in the export.
  // Keep exactly one node per element, in the document's chosen reading order.
  const content = c.order.map(part).join('');
  return `<section class="pg-hero pg-composed" id="${section.id}" data-section="${section.id}" data-composition="true" data-placement="${c.placement}"><div class="pg-composition-grid">${content}</div></section>`;
}
function placementStyles(selector: string, placement: ImagePlacement, c: Composition, mobile: boolean): string {
  // Slots for the three text elements are distinct grid rows, preserving their chosen ordering.
  const text = c.order.filter(name => name !== 'image');
  const inline = c.order.map(name => `"${name}"`).join(' ');
  let areas = text.map(name => `"${name}"`).join(' ');
  let columns = 'minmax(0,1fr)';
  if (placement === 'left' || placement === 'right') {
    columns = placement === 'left' ? 'minmax(0,1fr) minmax(0,1.45fr)' : 'minmax(0,1.45fr) minmax(0,1fr)';
    areas = text.map(name => placement === 'left' ? `"image ${name}"` : `"${name} image"`).join(' ');
  } else if (placement === 'above') areas = `"image" ${areas}`;
  else if (placement === 'below') areas += ' "image"';
  else if (placement === 'inline') areas = inline;
  const background = placement === 'background';
  const align = mobile && c.mobileAlign !== 'inherit' ? c.mobileAlign : c.align;
  const justify = align === 'left' ? 'start' : align === 'right' ? 'end' : 'center';
  const image = `${selector} .pg-part-image`;
  return `${selector} .pg-composition-grid{grid-template-columns:${columns};grid-template-areas:${areas};justify-items:${justify};text-align:${align};}
${selector} .pg-part:not(.pg-part-image){width:${mobile ? 100 : c.contentWidth}%;justify-self:${justify};}
${image}{position:${background ? 'absolute' : 'relative'};inset:${background ? '0' : 'auto'};grid-area:${background ? 'auto' : 'image'};z-index:${background ? '0' : '1'};pointer-events:${background ? 'none' : 'auto'};width:100%;height:${background ? '100%' : mobile ? '300px' : 'clamp(240px,38vw,480px)'};align-self:center;}
${image}>.pg-composition-visual{opacity:${c.imageOpacity};filter:blur(${c.blur}px);transform:${mobile ? 'none' : `translate(${c.offsetX}%,${c.offsetY}%) rotate(${c.rotation}deg) scale(${c.scale})`};}
${image}:after{content:'';position:absolute;inset:0;background:${c.overlay};opacity:${background ? c.overlayOpacity : 0};pointer-events:none;}
${image} .pg-media-frame,${image} .pg-image,${image} .pg-art{height:100%;width:100%;max-width:none;margin:0;}
${selector} .pg-composition-grid{min-height:${mobile ? 0 : c.minHeight - 100}px;}
${selector}{min-height:${mobile ? background ? 420 : 0 : c.minHeight}px;}
`;
}
export function compositionStyles(sections: Section[]): string {
  const composed = sections.filter(s => s.composition);
  if (!composed.length) return '';
  const base = `.pg-hero.pg-composed{display:block;position:relative;isolation:isolate;overflow:hidden;padding:50px clamp(24px,4vw,52px)}
.pg-composition-grid{display:grid;align-content:center;align-items:center;position:static;column-gap:32px}
.pg-composed .pg-part{min-width:0;position:relative;z-index:1;overflow-wrap:anywhere}
.pg-composed .pg-part-title{grid-area:title}.pg-composed .pg-part-description{grid-area:description}.pg-composed .pg-part-cta{grid-area:cta}
.pg-composed .pg-part h1,.pg-composed .pg-part p{margin:0;max-width:none}
.pg-composed .pg-part:empty{display:none}.pg-composed .pg-part-image{overflow:hidden;border-radius:var(--radius)}
.pg-composition-visual{position:absolute;inset:0;transform-origin:center;pointer-events:none}
.pg-composed .pg-part-image>.pg-composition-visual>.pg-art{height:100%;max-width:100%}
@media(max-width:640px){.pg-hero.pg-composed{padding:36px 24px}.pg-composed h1{font-size:clamp(32px,14vw,calc(57px * var(--scale)))}}
`;
  return base + composed.map(s => {
    const c = s.composition!, selector = `#${s.id}.pg-composed`;
    return `${selector} .pg-composition-grid{row-gap:${c.gap}px}` + placementStyles(selector, c.placement, c, false) +
      `@media(max-width:640px){${placementStyles(selector, effectivePlacement(c, true), c, true)}}`;
  }).join('');
}
