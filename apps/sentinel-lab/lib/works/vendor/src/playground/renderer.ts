import { type Project, onColor, safeHref, RENDERER } from "./model";
import { pageStyles } from "./pageStyles";
import { pageRuntime } from "./runtime";
import { glassScript, glassStyles, glassMarkup } from './glassBundle';
export const escapeHtml = (v: string) =>
  v.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
export function tokensCss(p: Project): string {
  const t = p.tokens,
    g = p.glass;
  return `:root{--bg:${t.background};--fg:${t.foreground};--accent:${t.accent};--space:${t.spacing}px;--radius:${t.radius}px;--scale:${t.typeScale};--width:${t.width}px;--display:${t.font === "editorial" ? "Georgia,serif" : "Arial,Helvetica,sans-serif"};--button-text:${onColor(t.accent)};--solid:${g.solid};--header-text:${onColor(g.solid)};--glass-radius:${g.radius}px;--frost:${g.frost}px}`;
}
export function art(): string {
  return `<svg class="pg-art" viewBox="0 0 400 500" aria-hidden="true">${Array.from({ length: 24 }, (_, i) => `<ellipse cx="200" cy="${250 - i * 2.2}" rx="${63 + i * 4.8}" ry="${87 + i * 5.9}"/>`).join("")}</svg>`;
}
export function pageMarkup(p: Project): string {
  const esc = escapeHtml;
  const image = (s: Project["sections"][number], logo = false) => {
    if(!s.image)return '';
    const img=`<img class="pg-image${logo?' pg-brand-image':''}" src="${esc(s.image)}" alt="${esc(s.media?.decorative?'':s.alt)}"${s.id==='hero'||logo?'':' loading="lazy"'}>`;
    return s.media ? `<span class="pg-media-frame${logo?' pg-logo-frame':''}">${img}<span class="pg-media-tint" aria-hidden="true"></span></span>` : img;
  };

  const visible = (id: string) =>
    p.sections.some((s) => s.id === id && s.visible);
  const button = (s: Project["sections"][number], outline = false) =>
    s.cta
      ? `<a class="${outline ? "pg-outline" : "pg-button"}" href="${esc(safeHref(s.href))}">${esc(s.cta)}</a>`
      : "";
  return `<a class="pg-skip" href="#page-main">Skip to content</a><div class="pg-wrap">${p.sections
    .filter((s) => s.visible || s.id === "header" || s.id === "footer")
    .map((s) => {
      const attrs = `id="${s.id}" data-section="${s.id}"`;
      if (s.id === "header" && !s.visible)
        return '<main id="page-main" tabindex="-1">';
      if (s.id === "footer" && !s.visible) return "</main>";
      if (s.id === "header")
        return `<header class="pg-header${p.rendererVersion === RENDERER ? ' ew-glass-header' : ''}" ${attrs}>${p.rendererVersion === RENDERER ? `<div class="ew-header__inner">${glassMarkup}` : '<canvas aria-hidden="true"></canvas>'}<a class="pg-logo ew-brand" href="#page-main">${s.image ? image(s,true) : esc(s.title)}</a><button class="pg-menu ew-menu-toggle" aria-label="Toggle navigation" aria-expanded="false" aria-controls="page-nav">Menu</button><nav class="pg-nav ew-nav" id="page-nav" aria-label="Page navigation">${s.description
          .split("|")
          .slice(0, 2)
          .map((v, i) =>
            visible(i ? "services" : "work")
              ? `<a href="#${i ? "services" : "work"}">${esc(v.trim())}</a>`
              : "",
          )
          .join(
            "",
          )}${button(s, true)}</nav>${p.rendererVersion === RENDERER ? '</div>' : ''}</header><main id="page-main" tabindex="-1">`;
      if (s.id === "hero")
        return `<section class="pg-hero ${s.layout}" ${attrs}><div class="pg-copy"><h1>${esc(s.title)}</h1><p>${esc(s.description)}</p>${button(s)}</div>${s.image ? image(s) : art()}</section>`;
      if (s.id === "services")
        return `<section class="pg-services" aria-label="${esc(s.title)}" ${attrs}>${s.image?`<div class="pg-section-media">${image(s)}</div>`:""}${s.description
          .split("\n")
          .filter(Boolean)
          .slice(0, 6)
          .map(
            (v, i) =>
              `<div class="pg-service"><span aria-hidden="true">0${i + 1}</span><h3>${esc(v)}</h3></div>`,
          )
          .join("")}</section>`;
      if (s.id === "work")
        return `<section class="pg-work" ${attrs}><h2 class="pg-section-title">${esc(s.title)}</h2><p>${esc(s.description)}</p>${s.image ? image(s) : `<div class="pg-studies"><figure class="pg-study"><div class="pg-study-art">${art()}</div><figcaption>Form study / 01</figcaption></figure><figure class="pg-study"><div class="pg-study-art">${art()}</div><figcaption>Color study / 02</figcaption></figure></div>`}</section>`;
      if (s.id === "contact")
        return `<section class="pg-contact" ${attrs}>${s.image?`<div class="pg-section-media">${image(s)}</div>`:""}<h2 class="pg-section-title">${esc(s.title)}</h2><div><p>${esc(s.description)}</p>${button(s)}</div></section>`;
      return `</main><footer class="pg-footer" ${attrs}><strong>${esc(s.title)}</strong><span>${esc(s.description)}</span></footer>`;
    })
    .join("")}</div>`;
}
export function runtimeConfig(
  p: Project,
  editing = false,
  selected = "",
  bridge = false,
) {
  return { glass: p.glass, editing, selected, bridge, originalGlass: p.rendererVersion === RENDERER };
}
export function runtimeScript(
  p: Project,
  editing = false,
  selected = "",
  bridge = false,
): string {
  return `${glassScript}\n(${pageRuntime.toString()})(${JSON.stringify(runtimeConfig(p, editing, selected, bridge)).replace(/</g, "\\u003c")});`;
}
function mediaStyles(p: Project) {
  const settings=p.sections.filter(s=>s.media&&s.image);if(!settings.length)return '';
  return `.pg-media-frame{display:block;position:relative;overflow:hidden;border-radius:var(--radius);min-width:0}.pg-media-frame>.pg-image{width:100%;height:100%;display:block}.pg-media-tint{position:absolute;inset:0;pointer-events:none}.pg-logo-frame{width:140px;height:40px;border-radius:0}.pg-brand-image{max-height:40px}.pg-services,.pg-contact{position:relative;isolation:isolate}.pg-section-media{position:absolute;inset:0;z-index:-1;overflow:hidden}.pg-section-media .pg-media-frame{height:100%}.pg-section-media .pg-image{height:100%}` + settings.map(s=>{const m=s.media!;return `#${s.id} .pg-media-frame>.pg-image{object-fit:${m.fit};object-position:${m.x}% ${m.y}%;transform:scale(${m.zoom})}#${s.id} .pg-media-tint{background:${m.tint};opacity:${m.opacity}}@media(max-width:640px){#${s.id} .pg-media-frame>.pg-image{object-position:${m.mobileX??m.x}% ${m.mobileY??m.y}%}}`;}).join('');
}
export function renderedStyles(p: Project) { return pageStyles + mediaStyles(p) + (p.rendererVersion === RENDERER ? glassStyles + `
.pg-header.ew-glass-header{padding:0;background:transparent;backdrop-filter:none;-webkit-backdrop-filter:none;isolation:auto;box-shadow:none}
.pg-header.ew-glass-header:before{display:none}
.pg-header .ew-header__inner{width:100%;display:flex;align-items:center;justify-content:space-between;padding:18px 28px;min-height:64px}
.pg-header canvas.ew-liquid-light{z-index:auto}
.pg-header .pg-nav.open{background:var(--solid)}
@media(max-width:640px){.pg-header .ew-header__inner{padding:14px 18px}.pg-header .pg-nav{position:absolute}}
` : ''); }
export function pageDocument(
  p: Project,
  options: {
    editing?: boolean;
    selected?: string;
    bridge?: boolean;
    external?: boolean;
  } = {},
): string {
  return `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="generator" content="${p.rendererVersion}"><title>${escapeHtml(p.name)}</title>${options.external ? '<link rel="stylesheet" href="tokens.css"><link rel="stylesheet" href="styles.css">' : `<style id="page-style">${tokensCss(p)}${renderedStyles(p)}</style>`}</head><body data-reduced="${p.glass.reducedMotion}"><div id="page-root">${pageMarkup(p)}</div>${options.external ? '<script src="script.js" defer></script>' : `<script>${runtimeScript(p, options.editing, options.selected, options.bridge)}</script>`}</body></html>`;
}
