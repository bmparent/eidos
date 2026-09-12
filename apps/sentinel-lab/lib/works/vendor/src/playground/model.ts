import { validateMedia, validateBrand, type MediaSettings, type BrandKit } from "./media";
import approved from "./approved-glass-preset.json";
import { validateComposition, type Composition } from './compositionSchema';
import { validateAuthoring, flattenNodes, BLOCK_LIMITS, type Authoring, type BlockNode } from './authoringSchema';
export const VERSION = 4;
export const LEGACY_RENDERER = "eidos-portable-glass-1.0.0";
export const RENDERER = "eidos-original-glass-2.0.0";
export const sectionTypes = [
  "header",
  "hero",
  "services",
  "work",
  "contact",
  "footer", "about", "faq", "gallery", "testimonials", "pricing", "content",
] as const;
export type SectionType = (typeof sectionTypes)[number];
export const labels: Record<SectionType, string> = {
  header: "Header",
  hero: "Hero",
  services: "Services",
  work: "Selected work",
  contact: "Contact",
  footer: "Footer", content: "Content section", about: "About", faq: "FAQ", gallery: "Gallery", testimonials: "Testimonials", pricing: "Services / pricing",
};
export type SectionStyle = { spacing: number; align: 'left'|'center'|'right'; background: string; mobileSpacing?: number; mobileAlign?: 'left'|'center'|'right' };
export type Card = {id:string;title:string;description:string;image:string;alt:string;media?:MediaSettings};
export const sectionKind = (s: Section): SectionType => s.type || s.id as SectionType;
export const mediaSlots = (p:Project): (Section|Card|BlockNode)[] => p.sections.flatMap(s=>[s,...(s.cards||[]),...(s.authoring?flattenNodes(s.authoring.root):[])]);
export function upgradeProject(p:Project):Project { return p.schemaVersion >= 3 ? p : validateProject({...p,schemaVersion:2,sections:p.sections.map(s=>({...s,type:sectionKind(s)}))}); }
export function newBlock(type:SectionType):Section { return {...makeSection(type,labels[type]),id:'block-'+crypto.randomUUID(),type,...(['faq','gallery','testimonials','pricing'].includes(type)?{cards:[]}: {})}; }
export type Section = {
  authoring?: Authoring;
  composition?: Composition;
  media?: MediaSettings;
  id: string;
  type?: SectionType;
  style?: SectionStyle;
  cards?: Card[];
  navigation?: {label:string;href:string}[];
  visible: boolean;
  title: string;
  description: string;
  cta: string;
  href: string;
  layout: "split" | "center";
  image: string;
  alt: string;
};
export type Project = {
  brand?: BrandKit;
  schemaVersion: 1 | 2 | 3 | 4;
  rendererVersion: string;
  template: "landing" | "homepage" | "portfolio" | "about" | "contact";
  name: string;
  tokens: {
    background: string;
    foreground: string;
    accent: string;
    spacing: number;
    radius: number;
    typeScale: number;
    font: "editorial" | "modern";
    width: number;
  };
  glass: {
    presetId: string;
    enabled: boolean;
    frost: number;
    edge: number;
    prism: number;
    touch: number;
    radius: number;
    response: number;
    solid: string;
    start: number;
    end: number;
    reducedMotion: boolean;
  };
  sections: Section[];
};
export const palettes = {
  Moss: { background: "#292e18", foreground: "#f4f4e7", accent: "#d5f87b" },
  Ink: { background: "#172632", foreground: "#eef4f7", accent: "#a8dcec" },
  Clay: { background: "#e9dfd0", foreground: "#352c25", accent: "#77563e" },
};
const makeSection = (
  id: SectionType,
  title: string,
  description = "",
  cta = "",
  href = "",
): Section => ({
  id,
  visible: true,
  title,
  description,
  cta,
  href,
  layout: "split",
  image: "",
  alt: "",
});
export function createProject(
  template: Project["template"] = "landing",
): Project {
  const p: Project = {
    schemaVersion: 1,
    rendererVersion: RENDERER,
    template,
    name: "Studio landing",
    tokens: {
      ...palettes.Moss,
      spacing: 76,
      radius: 20,
      typeScale: 1,
      font: "editorial",
      width: 1280,
    },
    glass: {
      presetId: approved.id,
      enabled: true,
      frost: approved.lens.frostPx,
      edge: approved.light.whitePeak,
      prism: approved.light.colourPeak,
      touch: approved.contact.pressScale,
      radius: approved.site.desktopRadiusPx,
      response: approved.springs.light,
      solid: approved.site.solidColour,
      start: approved.site.solidThroughPx,
      end: approved.site.fullyLiquidAtPx,
      reducedMotion: false,
    },
    sections: [
      makeSection("header", "FORMA", "Work|About", "Let’s talk ↗", "#contact"),
      makeSection(
        "hero",
        "Good things\ntake shape.",
        "Independent design for brands\nwith a point of view.",
        "Explore our work ↗",
        "#work",
      ),
      makeSection(
        "services",
        "What we bring.",
        "Brand identity\nDigital experiences\nCreative direction",
      ),
      makeSection(
        "work",
        "A different perspective.",
        "An exploration of form, color, and possibility.\nReplace these studies with your own work.",
      ),
      makeSection(
        "contact",
        "Have something\nin mind?",
        "We’d love to hear what you’re thinking.",
        "Start a conversation ↗",
        "mailto:hello@example.com",
      ),
      makeSection("footer", "FORMA", "Independent by design."),
    ],
  };
  if (template === "homepage") {
    p.tokens = {...p.tokens,...palettes.Ink,font:'modern',spacing:100};
    p.glass.solid = palettes.Ink.background;
    p.name = "Studio homepage";
    p.sections[1].title = "A little different.\nBy design.";
    p.sections[1].layout = "center";
  }
  if (template === "portfolio") {
    p.tokens = {...p.tokens,...palettes.Clay,spacing:120,radius:4};
    p.glass.solid = palettes.Clay.background;
    p.name = "Design portfolio";
    p.sections[1].title = "Thoughtful work.\nLasting impressions.";
    p.sections[2].visible = false;
  }
  if(template === 'about' || template === 'contact') {
    p.schemaVersion=2;p.name=template==='about'?'About page':'Contact page';p.sections=p.sections.map(s=>({...s,type:sectionKind(s)}));
    p.sections[1].title=template==='about'?'About your business':'Let us talk';p.sections[1].description='Add your own story and the details visitors need.';
    if(template==='about') p.sections.splice(2,0,{...newBlock('about'),title:'Our story',description:'Describe your business in your own words.'});
    if(template==='contact'){p.sections[2].visible=false;p.sections[3].visible=false;p.sections[4].description='Replace the sample email with your contact address. This link opens an email app; no form is connected.';}
  }
  return p;
}
const record = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== "object" || Array.isArray(v))
    throw new Error("Invalid project structure.");
  return v as Record<string, unknown>;
};
const string = (v: unknown, max = 2000) => {
  if (typeof v !== "string" || v.length > max)
    throw new Error("Invalid or oversized text field.");
  return v;
};
const number = (v: unknown, min: number, max: number) => {
  if (typeof v !== "number" || !Number.isFinite(v) || v < min || v > max)
    throw new Error(
      `A setting is outside its supported range (${min}–${max}).`,
    );
  return v;
};
const bool = (v: unknown) => {
  if (typeof v !== "boolean") throw new Error("Invalid switch value.");
  return v;
};
const choice = <T extends string>(v: unknown, values: readonly T[]): T => {
  if (!values.includes(v as T)) throw new Error("Unsupported project option.");
  return v as T;
};
const color = (v: unknown) => {
  const s = string(v, 7);
  if (!/^#[\da-f]{6}$/i.test(s))
    throw new Error("Colors must be six-digit hex values.");
  return s;
};
export function safeHref(value: string): string {
  if (
    /^(#[\w-]*|mailto:[^\s<>"']+|tel:\+?[\d ()-]+|https?:\/\/[^\s<>"']+)$/i.test(
      value,
    )
  )
    return value;
  return "#";
}
export function validateProject(input: unknown): Project {
  const p = record(input);
  if (![1, 2, 3, VERSION].includes(p.schemaVersion as number) || ![RENDERER, LEGACY_RENDERER].includes(String(p.rendererVersion)))
    throw new Error(
      "This project needs a different Playground version. Your current design is unchanged.",
    );
  if (p.schemaVersion === 4) {
    if (new TextEncoder().encode(JSON.stringify(p)).length > BLOCK_LIMITS.documentBytes) throw Error('This design exceeds the 16 MB local authoring limit. Your current design is unchanged.');
    if (Object.keys(p).some(k=>!['schemaVersion','rendererVersion','template','name','tokens','glass','sections','brand'].includes(k))) throw Error('Unsupported project field.');
  }
  const t = record(p.tokens),
    g = record(p.glass);
  if (!Array.isArray(p.sections) || (p.schemaVersion===1 ? p.sections.length!==6 : p.sections.length<3 || p.sections.length>16))
    throw new Error("Version 1 requires six sections; versions 2 and 3 support 3 to 16.");
  const seen = new Set<string>();
  const sections = p.sections.map((value) => {
    const s = record(value),
      id = p.schemaVersion===1 ? choice(s.id, sectionTypes.slice(0,6)) : string(s.id,80);
    if(!/^[a-z][a-z0-9-]*$/.test(id)||["page-main","page-nav","page-root"].includes(id))throw Error("Invalid or reserved section identity.");
    const type = p.schemaVersion===1 ? id as SectionType : choice(s.type,sectionTypes);
    if (s.composition !== undefined && (![3,4].includes(p.schemaVersion as number) || type !== 'hero')) throw Error('Spatial composition requires a version 3 hero.');
    if (p.schemaVersion === 3 && type === 'hero' && s.composition === undefined) throw Error('Version 3 heroes require composition settings.');
    if (s.authoring !== undefined && (p.schemaVersion !== 4 || !['hero','content'].includes(type))) throw Error('Responsive blocks require a version 4 hero or content section.');
    if (p.schemaVersion === 4 && type === 'hero' && s.authoring === undefined) throw Error('Version 4 heroes require responsive blocks.');
    if (type === 'content' && (p.schemaVersion !== 4 || s.authoring === undefined)) throw Error('A content section requires responsive blocks.');
    if (p.schemaVersion === 4 && Object.keys(s).some(k=>!['id','type','visible','title','description','cta','href','layout','image','alt','style','cards','navigation','media','composition','authoring'].includes(k))) throw Error('Unsupported section field.');
    if (seen.has(id)) throw new Error("Duplicate section.");
    seen.add(id);
    const image = string(s.image, 4_500_000);
    if (
      image &&
      !/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+=*$/.test(image)
    )
      throw new Error("Images must be embedded PNG, JPEG, or WebP files.");
    if (image) {
      const encoded = image.slice(image.indexOf(",") + 1);
      let bytes: string;
      try { bytes = atob(encoded); } catch { throw new Error("Image data is malformed. Your current design is unchanged."); }
      const valid = image.startsWith("data:image/png;") ? bytes.startsWith("\x89PNG\r\n\x1a\n")
        : image.startsWith("data:image/jpeg;") ? bytes.startsWith("\xff\xd8\xff")
          : bytes.startsWith("RIFF") && bytes.slice(8, 12) === "WEBP";
      if (!valid || btoa(bytes) !== encoded) throw new Error("Image data does not match its image type.");
    }
    const href = string(s.href, 1000);
    if (href && safeHref(href) === "#" && href !== "#")
      throw new Error(
        "Use a full https:// link, an anchor, mailto:, or tel: destination.",
      );
    return {
      id,
      ...(p.schemaVersion!==1 ? {type,...validateStructure(s)} : {}),
      ...(s.composition === undefined ? {} : {composition:validateComposition(s.composition)}),
      ...(s.authoring === undefined ? {} : {authoring:validateAuthoring(s.authoring)}),
      visible: bool(s.visible),
      title: string(s.title, 240),
      description: string(s.description),
      cta: string(s.cta, 100),
      href,
      layout: choice(s.layout, ["split", "center"] as const),
      image,
      alt: string(s.alt, 300),
      ...(s.media === undefined ? {} : {media: validateMedia(s.media)}),
    };
  });
  if (sectionKind(sections[0]) !== "header" || sectionKind(sections[sections.length-1]) !== "footer" || sections.filter(s=>sectionKind(s)==="header").length!==1 || sections.filter(s=>sectionKind(s)==="footer").length!==1 || sections.filter(s=>sectionKind(s)==="hero").length!==1)
    throw new Error("Header and footer must stay at the ends of the page.");
  const nodes=sections.flatMap(s=>s.authoring?flattenNodes(s.authoring.root):[]);
  if(nodes.length>BLOCK_LIMITS.nodes)throw Error('This design supports at most 96 responsive nodes.');
  const identities=sections.flatMap(s=>[s.id,...(s.cards||[]).map(c=>c.id),...(s.authoring?flattenNodes(s.authoring.root).map(n=>n.id):[])]);
  if(new Set(identities).size!==identities.length)throw Error("Duplicate page identity.");
  return {
    schemaVersion: p.schemaVersion as 1|2|3|4,
    ...(p.brand === undefined ? {} : {brand:validateBrand(p.brand)}),
    rendererVersion: String(p.rendererVersion),
    template: choice(p.template, ["landing", "homepage", "portfolio", "about", "contact"] as const),
    name: string(p.name, 100),
    tokens: {
      background: color(t.background),
      foreground: color(t.foreground),
      accent: color(t.accent),
      spacing: number(t.spacing, 32, 128),
      radius: number(t.radius, 0, 48),
      typeScale: number(t.typeScale, 0.75, 1.3),
      width: number(t.width, 960, 1600),
      font: choice(t.font, ["editorial", "modern"] as const),
    },
    glass: {
      presetId: choice(g.presetId, [approved.id]),
      enabled: bool(g.enabled),
      frost: number(g.frost, 0, 16),
      edge: number(g.edge, 0, 1),
      prism: number(g.prism, 0, 0.8),
      touch: number(g.touch, 0, 6),
      radius: number(g.radius, 0, 48),
      response: number(g.response, 10, 60),
      solid: color(g.solid),
      start: number(g.start, 0, 32),
      end: number(g.end, 40, 200),
      reducedMotion: bool(g.reducedMotion),
    },
    sections,
  };
}
function validateStructure(s:Record<string,unknown>):Pick<Section,'style'|'cards'|'navigation'> {
 const out:Pick<Section,'style'|'cards'|'navigation'>={};
 if(s.style!==undefined){const v=record(s.style);out.style={spacing:number(v.spacing,0,160),align:choice(v.align,['left','center','right']),background:v.background===''?'':color(v.background),...(v.mobileSpacing===undefined?{}:{mobileSpacing:number(v.mobileSpacing,0,160)}),...(v.mobileAlign===undefined?{}:{mobileAlign:choice(v.mobileAlign,['left','center','right'] as const)})};}
 if(s.navigation!==undefined){if(!Array.isArray(s.navigation)||s.navigation.length>6)throw Error('Navigation supports at most six links.');out.navigation=s.navigation.map(v=>{const n=record(v),href=string(n.href,1000);if(href&&safeHref(href)==='#'&&href!=='#')throw Error('Unsafe navigation destination.');return {label:string(n.label,60),href};});}
 if(s.cards!==undefined){if(!Array.isArray(s.cards)||s.cards.length>12)throw Error('A block supports up to twelve cards.');const seen=new Set<string>();out.cards=s.cards.map(v=>{const c=record(v),id=string(c.id,80);if(!/^card-[a-z0-9-]+$/.test(id)||seen.has(id))throw Error('Invalid or duplicate card identity.');seen.add(id);const fixture=createProject();fixture.sections[1]={...fixture.sections[1],image:c.image as string,alt:c.alt as string,...(c.media===undefined?{}:{media:c.media as MediaSettings})};const valid=validateProject(fixture).sections[1];return {id,title:string(c.title,240),description:string(c.description,2000),image:valid.image,alt:valid.alt,...(valid.media?{media:valid.media}:{})};});}
 return out;
}
export function contrast(a: string, b: string): number {
  const luminance = (hex: string) => {
    const c = [1, 3, 5]
      .map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
  };
  const x = luminance(a),
    y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
export const onColor = (hex: string) =>
  contrast(hex, "#111111") > contrast(hex, "#ffffff") ? "#111111" : "#ffffff";
export function projectWarnings(p: Project): string[] {
  const out: string[] = [];
  if (!p.sections.find((s) => sectionKind(s) === "hero")?.visible)
    out.push(
      "The hero is hidden. Add a visible page heading before publishing.",
    );
  if (contrast(p.tokens.background, p.tokens.foreground) < 4.5)
    out.push(
      "Page text contrast is below 4.5:1. Adjust the background or text color.",
    );
  for (const s of p.sections.filter((s) => s.visible)) {
    if (s.composition && (s.composition.placement === 'background' || s.composition.mobilePlacement === 'background'))
      out.push('Review hero text over the actual image at desktop and mobile sizes. An overlay does not guarantee sufficient contrast.');
    if (s.image && !s.alt.trim() && !s.media?.decorative)
      out.push(`${labels[sectionKind(s)]} image needs a description.`);
    if (
      s.href.startsWith("#") &&
      s.href !== "#" &&
      !p.sections.some(
        (target) => target.id === s.href.slice(1) && target.visible,
      )
    )
      out.push(`${labels[sectionKind(s)]} links to a hidden section.`);
    if (s.href.includes("example.com"))
      out.push("Replace the sample contact email before publishing.");
  }
  for(const s of p.sections.filter(s=>s.visible)) {
    for(const link of [...(s.navigation||[]),...(s.cta?[{label:s.cta,href:s.href}]:[])]) {
      if(!link.label.trim())out.push('A navigation link needs an accessible label.');
      if(!link.href||link.href==='#'||link.href.includes('example.com'))out.push('Replace unresolved links and sample destinations.');
      if(link.href.startsWith('#')&&link.href!=='#'&&!p.sections.some(t=>t.visible&&t.id===link.href.slice(1))&&link.href!=='#page-main')out.push('A navigation destination is missing or hidden.');
    }
    for(const item of [s,...(s.cards||[])]) {
      if(item.image&&!item.alt.trim()&&!item.media?.decorative)out.push('An image needs alternative text or decorative classification.');
      if(/\S{50,}/.test(item.title+' '+item.description))out.push('Long unbroken text may overflow on small screens.');
      if(/replace|your own|sample|placeholder/i.test(item.title+' '+item.description))out.push('Review remaining placeholder copy before publishing.');
    }
    if(['testimonials','pricing','gallery','faq'].includes(sectionKind(s))&&!s.cards?.length)out.push(labels[sectionKind(s)]+' is empty; add your own content.');
    if(sectionKind(s)==='contact')out.push('Contact links are not a connected form. Review the destination and test it.');
  }
  return [...new Set(out)];
}
