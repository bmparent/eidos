import approved from "./approved-glass-preset.json";
export const VERSION = 1;
export const LEGACY_RENDERER = "eidos-portable-glass-1.0.0";
export const RENDERER = "eidos-original-glass-2.0.0";
export const sectionTypes = [
  "header",
  "hero",
  "services",
  "work",
  "contact",
  "footer",
] as const;
export type SectionType = (typeof sectionTypes)[number];
export const labels: Record<SectionType, string> = {
  header: "Header",
  hero: "Hero",
  services: "Services",
  work: "Selected work",
  contact: "Contact",
  footer: "Footer",
};
export type Section = {
  id: SectionType;
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
  schemaVersion: 1;
  rendererVersion: string;
  template: "landing" | "homepage" | "portfolio";
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
  if (p.schemaVersion !== VERSION || ![RENDERER, LEGACY_RENDERER].includes(String(p.rendererVersion)))
    throw new Error(
      "This project needs a different Playground version. Your current design is unchanged.",
    );
  const t = record(p.tokens),
    g = record(p.glass);
  if (!Array.isArray(p.sections) || p.sections.length !== 6)
    throw new Error("Project must contain its six supported sections.");
  const seen = new Set<string>();
  const sections = p.sections.map((value) => {
    const s = record(value),
      id = choice(s.id, sectionTypes);
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
      visible: bool(s.visible),
      title: string(s.title, 240),
      description: string(s.description),
      cta: string(s.cta, 100),
      href,
      layout: choice(s.layout, ["split", "center"] as const),
      image,
      alt: string(s.alt, 300),
    };
  });
  if (sections[0].id !== "header" || sections[5].id !== "footer")
    throw new Error("Header and footer must stay at the ends of the page.");
  return {
    schemaVersion: 1,
    rendererVersion: String(p.rendererVersion),
    template: choice(p.template, ["landing", "homepage", "portfolio"] as const),
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
  if (!p.sections.find((s) => s.id === "hero")?.visible)
    out.push(
      "The hero is hidden. Add a visible page heading before publishing.",
    );
  if (contrast(p.tokens.background, p.tokens.foreground) < 4.5)
    out.push(
      "Page text contrast is below 4.5:1. Adjust the background or text color.",
    );
  for (const s of p.sections.filter((s) => s.visible)) {
    if (s.image && !s.alt.trim())
      out.push(`${labels[s.id]} image needs a description.`);
    if (
      s.href.startsWith("#") &&
      s.href !== "#" &&
      !p.sections.some(
        (target) => target.id === s.href.slice(1) && target.visible,
      )
    )
      out.push(`${labels[s.id]} links to a hidden section.`);
    if (s.href.includes("example.com"))
      out.push("Replace the sample contact email before publishing.");
  }
  return [...new Set(out)];
}
