import type { Project } from "./model";
type RuntimeConfig = {
  glass: Project["glass"];
  editing: boolean;
  selected: string;
  bridge: boolean;
  originalGlass: boolean;
  channel?: string;
  revision?: number;
};
declare const EidosGlass: { mount: (glass: Project['glass']) => () => void };
// Deliberately self-contained: the exact compiled function also runs in standalone exports.
export function pageRuntime(initial: RuntimeConfig) {
  let config = initial;
  let dispose = () => {};
  let headerSource = document.querySelector('.pg-header')?.outerHTML || '';
  // Patch committed content by stable identities. Editor chrome is removed by
  // playground-before-update before comparison; unchanged nodes retain focus/scroll.
  function patch(target: Element, source: Element) {
    for (const attr of Array.from(target.attributes)) if (!source.hasAttribute(attr.name)) target.removeAttribute(attr.name);
    for (const attr of Array.from(source.attributes)) if (target.getAttribute(attr.name) !== attr.value) target.setAttribute(attr.name, attr.value);
    let cursor: ChildNode | null = target.firstChild;
    for (const desired of Array.from(source.childNodes)) {
      const key = desired instanceof Element ? desired.id || desired.getAttribute('data-composition-part') : null;
      let current = cursor;
      if (key) current = Array.from(target.childNodes).find(n => n instanceof Element && (n.id || n.getAttribute('data-composition-part')) === key) || null;
      if (!current || current.nodeName !== desired.nodeName) {
        const clone = desired.cloneNode(true); target.insertBefore(clone, cursor); cursor = clone.nextSibling; continue;
      }
      if (current !== cursor) target.insertBefore(current, cursor);
      if (current instanceof Element && desired instanceof Element) {
        // Glass owns its own descendants. Do not reconcile its live canvases/SVG.
        if (!current.matches('.ew-glass-header')) patch(current, desired);
      } else if (current.nodeValue !== desired.nodeValue) current.nodeValue = desired.nodeValue;
      cursor = current.nextSibling;
    }
    while (cursor) { const next = cursor.nextSibling; cursor.remove(); cursor = next; }
  }
  function mount() {
    dispose();
    const disposeGlass = config.originalGlass ? EidosGlass.mount(config.glass) : () => {};
    const header = config.originalGlass ? null : document.querySelector<HTMLElement>(".pg-header");
    const canvas = header?.querySelector<HTMLCanvasElement>("canvas");
    const context = canvas?.getContext("2d");
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    const abort = new AbortController(),
      options = { signal: abort.signal };
    const g = config.glass;
    let frame = 0,
      last = 0,
      active = false,
      pressed = false,
      targetX = 0,
      targetY = 0,
      x = 0,
      y = 0,
      vx = 0,
      vy = 0,
      light = 0,
      lv = 0,
      press = 0,
      pv = 0,
      material = 0;
    const motion = () => !reduced.matches && !g.reducedMotion;
    const smooth = (n: number) => {
      const t = Math.max(0, Math.min(1, n));
      return t * t * (3 - 2 * t);
    };
    const spring = (
      v: number,
      velocity: number,
      target: number,
      frequency: number,
      dt: number,
    ) => {
      const d = v - target,
        j = velocity + frequency * d,
        e = Math.exp(-frequency * dt);
      return [target + (d + j * dt) * e, (velocity - frequency * j * dt) * e];
    };
    function draw(time: number) {
      frame = 0;
      const dt = Math.min((time - (last || time - 16)) / 1000, 0.04);
      last = time;
      [x, vx] = spring(x, vx, targetX, 48, dt);
      [y, vy] = spring(y, vy, targetY, 48, dt);
      [light, lv] = spring(
        light,
        lv,
        active && motion() ? 1 : 0,
        g.response,
        dt,
      );
      [press, pv] = spring(press, pv, pressed && motion() ? 1 : 0, 32, dt);
      if (header && canvas && context) {
        const { width: w, height: h } = header.getBoundingClientRect();
        const dpr = Math.min(devicePixelRatio || 1, 1.5);
        if (
          canvas.width !== Math.round(w * dpr) ||
          canvas.height !== Math.round(h * dpr)
        ) {
          canvas.width = Math.round(w * dpr);
          canvas.height = Math.round(h * dpr);
        }
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        context.clearRect(0, 0, w, h);
        if (material > 0 && light > 0.001) {
          const radius = Math.min(g.radius, h / 2, w / 2),
            perimeter = 2 * (w + h - 4 * radius) + 2 * Math.PI * radius;
          // Rounded rectangle samples with local normals keep highlights attached to the rim.
          for (let t = 0; t < perimeter; t += 2) {
            let d = t,
              px: number,
              py: number,
              nx = 0,
              ny = -1;
            const lengths = [
              w - 2 * radius,
              (Math.PI * radius) / 2,
              h - 2 * radius,
              (Math.PI * radius) / 2,
              w - 2 * radius,
              (Math.PI * radius) / 2,
              h - 2 * radius,
              (Math.PI * radius) / 2,
            ];
            let edge = 0;
            while (edge < 7 && d > lengths[edge]) d -= lengths[edge++];
            if (edge === 0) {
              px = radius + d;
              py = 0;
            } else if (edge === 2) {
              px = w;
              py = radius + d;
              nx = 1;
              ny = 0;
            } else if (edge === 4) {
              px = w - radius - d;
              py = h;
              nx = 0;
              ny = 1;
            } else if (edge === 6) {
              px = 0;
              py = h - radius - d;
              nx = -1;
              ny = 0;
            } else {
              const corner = (edge - 1) / 2,
                angle =
                  -Math.PI / 2 +
                  (corner * Math.PI) / 2 +
                  (radius ? d / radius : 0);
              nx = Math.cos(angle);
              ny = Math.sin(angle);
              px = (corner < 2 ? w - radius : radius) + radius * nx;
              py =
                (corner === 0 || corner === 3 ? radius : h - radius) +
                radius * ny;
            }
            const dx = x - px,
              dy = y - py,
              distance = Math.hypot(dx, dy);
            const orientation =
              0.3 + 0.7 * Math.abs((dx * nx + dy * ny) / Math.max(distance, 1));
            const power =
              Math.exp((-distance * distance) / (2 * 115 * 115)) *
              orientation *
              light *
              material *
              (1 + press * 0.18);
            if (power < 0.002) continue;
            const deformation = press * g.touch * power * 0.3;
            context.fillStyle = `rgba(255,255,255,${power * g.edge})`;
            context.beginPath();
            context.arc(
              px - nx * (0.9 + deformation),
              py - ny * (0.9 + deformation),
              0.85,
              0,
              Math.PI * 2,
            );
            context.fill();
            const hue = 180 + 110 * Math.sin((dx * -ny + dy * nx) / 48);
            context.fillStyle = `hsla(${hue},95%,75%,${power * g.prism})`;
            context.beginPath();
            context.arc(
              px - nx * (2.8 + deformation),
              py - ny * (2.8 + deformation),
              1.35,
              0,
              Math.PI * 2,
            );
            context.fill();
          }
        }
      }
      if (
        Math.abs(x - targetX) +
          Math.abs(y - targetY) +
          Math.abs(lv) +
          Math.abs(pv) >
          0.02 ||
        Math.abs(light - (active && motion() ? 1 : 0)) > 0.002
      )
        schedule();
    }
    function schedule() {
      if (!frame && !document.hidden) frame = requestAnimationFrame(draw);
    }
    function scroll() {
      material = g.enabled
        ? smooth((scrollY - g.start) / (g.end - g.start))
        : 0;
      header?.style.setProperty("--liquid", String(material));
      schedule();
    }
    function reset() {
      active = false;
      pressed = false;
      schedule();
    }
    function pointer(event: PointerEvent) {
      if (!header || !motion()) return;
      const r = header.getBoundingClientRect();
      targetX = event.clientX - r.left;
      targetY = event.clientY - r.top;
      const distance = Math.hypot(
        Math.max(0, -targetX, targetX - r.width),
        Math.max(0, -targetY, targetY - r.height),
      );
      active = distance < 170;
      if (last === 0) {
        x = targetX;
        y = targetY;
      }
      schedule();
    }
    document.addEventListener("pointermove", pointer, {
      ...options,
      passive: true,
    });
    header?.addEventListener(
      "pointerdown",
      (e) => {
        pointer(e);
        pressed = true;
        schedule();
      },
      options,
    );
    window.addEventListener(
      "pointerup",
      () => {
        pressed = false;
        schedule();
      },
      options,
    );
    window.addEventListener("pointercancel", reset, options);
    window.addEventListener("blur", reset, options);
    document.addEventListener("pointerleave", reset, options);
    document.addEventListener(
      "visibilitychange",
      () => {
        reset();
        if (document.hidden) {
          cancelAnimationFrame(frame);
          frame = 0;
          last = 0;
        } else scroll();
      },
      options,
    );
    window.addEventListener("scroll", scroll, { ...options, passive: true });
    window.addEventListener("pageshow", scroll, options);
    reduced.addEventListener("change", reset, options);
    const observer = new ResizeObserver(() => schedule());
    if (header) observer.observe(header);
    const menu = document.querySelector<HTMLButtonElement>(".pg-menu");
    const nav = document.querySelector<HTMLElement>(".pg-nav");
    const closeMenu = () => {
      nav?.classList.remove("open");
      nav?.classList.remove("is-open");
      menu?.setAttribute("aria-expanded", "false");
    };
    menu?.addEventListener(
      "click",
      () => {
        const open = nav?.classList.toggle("open");
        nav?.classList.toggle("is-open", !!open);
        menu.setAttribute("aria-expanded", String(!!open));
      },
      options,
    );
    document.addEventListener(
      "keydown",
      (e) => {
        if (e.key === "Escape") {
          closeMenu();
          menu?.focus();
        }
      },
      options,
    );
    document.addEventListener(
      "click",
      (e) => {
        const target = e.target as HTMLElement;
        const section = target.closest<HTMLElement>("[data-section]");
        if (config.editing && config.bridge) {
          e.preventDefault();
          if (section)
            parent.postMessage(
              { type: "playground-select", channel: config.channel, id: section.dataset.section },
              "*",
            );
          return;
        }
        const link = target.closest<HTMLAnchorElement>("a");
        if (link) {
          closeMenu();
          const href = link.getAttribute("href") || "";
          if (config.bridge && href.startsWith("#")) {
            e.preventDefault();
            document
              .getElementById(href.slice(1) || "page-main")
              ?.scrollIntoView({
                behavior: motion() ? "smooth" : "instant",
                block: "start",
              });
          }
          if (config.bridge && !href.startsWith("#")) {
            e.preventDefault();
            parent.postMessage(
              { type: "playground-link", channel: config.channel, href: link.getAttribute("href") },
              "*",
            );
          }
        }
      },
      options,
    );
    document.body.dataset.editing = String(config.editing);
    document.body.dataset.reduced = String(g.reducedMotion);
    document
      .querySelectorAll("[data-section]")
      .forEach((s) =>
        s.classList.toggle(
          "pg-selected",
          s.getAttribute("data-section") === config.selected && config.editing,
        ),
      );
    scroll();
    dispose = () => {
      disposeGlass();
      abort.abort();
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }
  mount();
  if (initial.bridge) {
    window.addEventListener("message", (e) => {
      if (e.source !== parent || !e.data || typeof e.data !== "object") return;
      if (e.data.channel !== initial.channel) return;
      if (e.data.type === 'playground-selection' && typeof e.data.selected === 'string') {
        config.selected = e.data.selected;
        document.querySelectorAll('[data-section]').forEach(s => s.classList.toggle('pg-selected', config.editing && s.getAttribute('data-section') === config.selected));
        return;
      }
      if (
        e.data.type === "playground-update" &&
        typeof e.data.html === "string" &&
        typeof e.data.css === "string"
      ) {
        if (!e.data.config || e.data.config.channel !== initial.channel || !Number.isSafeInteger(e.data.config.revision) || e.data.config.revision <= (config.revision || 0)) return;
        // Incompatible document revisions cancel before any DOM can be replaced.
        window.dispatchEvent(new Event('playground-before-update'));
        const y = scrollY;
        const oldHeader = document.querySelector('.pg-header');
        const template = document.createElement('template'); template.innerHTML = e.data.html;
        const nextHeader = template.content.querySelector('.pg-header');
        const glassChanged = JSON.stringify(config.glass) !== JSON.stringify(e.data.config.glass) || config.originalGlass !== e.data.config.originalGlass;
        // Compare source header markup, never the glass runtime's live descendants.
        const headerMarkup = nextHeader?.outerHTML || '';
        const headerChanged = headerMarkup !== headerSource;
        if (glassChanged || headerChanged) dispose();
        const pageStyle = document.getElementById('page-style')!;
        if (pageStyle.textContent !== e.data.css) pageStyle.textContent = e.data.css;
        const pageRoot = document.getElementById('page-root')!;
        const incoming = document.createElement('div'); incoming.id = 'page-root'; incoming.append(template.content);
        patch(pageRoot, incoming);
        if (headerChanged && oldHeader?.isConnected && nextHeader) oldHeader.replaceWith(nextHeader.cloneNode(true));
        headerSource = headerMarkup;
        config = e.data.config;
        if (glassChanged || headerChanged) mount();
        document.body.dataset.editing = String(config.editing);
        document.body.dataset.reduced = String(config.glass.reducedMotion);
        window.scrollTo(0, y);
      }
      if (e.data.type === "playground-focus" && typeof e.data.id === "string")
        document
          .getElementById(e.data.id)
          ?.scrollIntoView({ behavior: "instant", block: "start" });
    });
    parent.postMessage({ type: "playground-ready", channel: initial.channel }, "*");
  }
}
