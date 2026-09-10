/** Self-contained editor-only interaction layer, stringified into the sandbox preview.
 * It is deliberately excluded from standalone exports. Only handles suppress touch scrolling.
 */
export function compositionRuntime(initial: { editing: boolean; bridge: boolean; compositionStamp?: string }) {
  if (!initial.bridge) return;
  let config = initial;
  let cleanup = () => {};
  let selected = '';
  function mount() {
    cleanup();
    if (!config.editing || !config.compositionStamp) return;
    const abort = new AbortController();
    const options = { signal: abort.signal };
    const roots = Array.from(document.querySelectorAll<HTMLElement>('[data-composition="true"]'));
    if (!roots.length) { cleanup = () => abort.abort(); return; }
    const style = document.createElement('style');
    style.dataset.compositionEditor = 'true';
    style.textContent = `
      .pg-part[data-composition-part]{outline-offset:4px}
      .pg-part.pg-compose-empty{min-height:44px}
      .pg-part.pg-part-selected{outline:2px solid #bce886}
      .pg-compose-handle{position:absolute;top:0;right:0;z-index:4;background:#15242c;color:#f5f8f4;border:1px solid #bce886;border-radius:7px;font:600 12px/1.2 Arial,sans-serif;min-width:44px;min-height:32px;padding:7px 10px;touch-action:none;cursor:grab;pointer-events:auto;box-shadow:0 2px 6px #0004}
      .pg-compose-handle:focus-visible{outline:3px solid #fff;outline-offset:2px}
      .pg-composed>.pg-compose-handle{top:10px;right:10px;z-index:4}
      .pg-compose-hud{position:fixed;z-index:9999;bottom:14px;left:50%;transform:translateX(-50%);max-width:calc(100vw - 24px);width:max-content;background:#15242c;color:#fff;border:1px solid #bce886;border-radius:12px;font:600 13px/1.5 Arial,sans-serif;padding:12px 16px;pointer-events:none;box-shadow:0 8px 30px #0006}
      .pg-compose-target{outline:3px dashed #bce886;outline-offset:-5px}
      .pg-compose-before{box-shadow:0 -4px #bce886!important}.pg-compose-after{box-shadow:0 4px #bce886!important}
      .pg-compose-dragging{cursor:grabbing!important;user-select:none}
    `;
    document.head.append(style);
    const handles: HTMLButtonElement[] = [];
    const emptyParts: HTMLElement[] = [];
    let hud: HTMLElement | null = null;
    let gesture: { pointer: number; x: number; y: number; part: string; root: HTMLElement; handle: HTMLButtonElement; stamp: string; moved: boolean } | null = null;
    type Destination = { placement: string } | { before: string | null };
    let destination: Destination | null = null;
    const labels: Record<string, string> = { title: 'Heading', description: 'Description', cta: 'Button', image: 'Image' };
    function send(action: Record<string, unknown>, stamp = config.compositionStamp) {
      parent.postMessage({ type: 'playground-composition', stamp, ...action }, '*');
    }
    function clearMarks() {
      document.querySelectorAll('.pg-compose-target,.pg-compose-before,.pg-compose-after').forEach(el => el.classList.remove('pg-compose-target', 'pg-compose-before', 'pg-compose-after'));
    }
    function reset() {
      const old = gesture;
      gesture = null;
      destination = null;
      hud?.remove(); hud = null;
      document.body.classList.remove('pg-compose-dragging');
      clearMarks();
      if (old?.handle.hasPointerCapture(old.pointer)) old.handle.releasePointerCapture(old.pointer);
    }
    function hint(text: string) {
      if (!hud) { hud = document.createElement('div'); hud.className = 'pg-compose-hud'; hud.setAttribute('role', 'status'); document.body.append(hud); }
      if (hud.textContent !== text) hud.textContent = text;
    }
    function locate(root: HTMLElement, x: number, y: number, part: string): Destination | null {
      clearMarks();
      const rect = root.getBoundingClientRect();
      if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) { hint('Release outside to cancel · Esc cancels'); return null; }
      const nx = (x - rect.left) / rect.width, ny = (y - rect.top) / rect.height;
      let placement: string | null = null;
      if (part === 'image') {
        if (ny < .13) placement = 'above';
        else if (ny > .87) placement = 'below';
        else if (innerWidth > 640 && nx < .17) placement = 'left';
        else if (innerWidth > 640 && nx > .83) placement = 'right';
        // The center band is a deliberate depth target, distinct from inline insertion.
        else if (nx > .4 && nx < .6 && ny > .39 && ny < .61) placement = 'background';
      }
      if (placement) {
        root.classList.add('pg-compose-target');
        hint(placement === 'background' ? 'Image → behind hero text' : `Image → ${placement} of hero content`);
        return { placement };
      }
      const targets = Array.from(root.querySelectorAll<HTMLElement>('[data-composition-part]')).filter(el => el.dataset.compositionPart !== part && el.dataset.compositionPart !== 'image');
      const target = targets.find(el => { const r = el.getBoundingClientRect(); return y <= r.top + r.height / 2; });
      if (target) {
        target.classList.add('pg-compose-before');
        hint(`${labels[part]} → before ${labels[target.dataset.compositionPart!]}`);
        return { before: target.dataset.compositionPart! };
      }
      const last = targets.at(-1);
      last?.classList.add('pg-compose-after');
      hint(`${labels[part]} → after content`);
      return { before: null };
    }
    for (const root of roots) {
      for (const el of root.querySelectorAll<HTMLElement>('[data-composition-part]')) {
        const part = el.dataset.compositionPart!;
        if (part !== 'image' && !el.textContent?.trim() && !el.querySelector('a,img')) {
          el.classList.add('pg-compose-empty'); emptyParts.push(el);
        }
        if (part === selected) el.classList.add('pg-part-selected');
        const handle = document.createElement('button');
        handle.type = 'button'; handle.className = 'pg-compose-handle'; handle.textContent = `↕ ${labels[part]}`;
        handle.setAttribute('aria-label', `Move ${labels[part].toLowerCase()}. Drag, or use Layout controls.`);
        // A background image is non-interactive; its editor handle belongs to the section instead.
        if (part === 'image') root.append(handle); else el.append(handle);
        handles.push(handle);
        handle.addEventListener('click', event => { event.preventDefault(); event.stopPropagation(); selected = part; send({ action: 'select', sectionId: root.id, part }); }, options);
        handle.addEventListener('pointerdown', event => {
          if (!event.isPrimary || event.button !== 0) return;
          event.preventDefault(); event.stopPropagation(); reset();
          handle.focus({ preventScroll: true });
          root.querySelectorAll('.pg-part-selected').forEach(v => v.classList.remove('pg-part-selected'));
          el.classList.add('pg-part-selected'); selected = part;
          gesture = { pointer: event.pointerId, x: event.clientX, y: event.clientY, root, part, handle, stamp: config.compositionStamp!, moved: false };
          handle.setPointerCapture(event.pointerId);
          hint(part === 'image' ? 'Drag to center: behind text · edges: beside/above/below · text: insert' : 'Drag between text elements · Esc cancels');
        }, options);
      }
    }
    document.addEventListener('pointermove', event => {
      if (!gesture || gesture.pointer !== event.pointerId) return;
      if (!gesture.moved && Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y) < 6) return;
      event.preventDefault(); gesture.moved = true;
      document.body.classList.add('pg-compose-dragging');
      destination = locate(gesture.root, event.clientX, event.clientY, gesture.part);
    }, { ...options, passive: false });
    document.addEventListener('pointerup', event => {
      if (!gesture || gesture.pointer !== event.pointerId) return;
      const g = gesture, d = destination;
      reset();
      if (g.moved && d) send({ action: 'move', sectionId: g.root.id, part: g.part, destination: d, mobile: innerWidth <= 640 }, g.stamp);
    }, options);
    document.addEventListener('pointercancel', reset, options);
    document.addEventListener('lostpointercapture', () => { if (gesture) reset(); }, options);
    document.addEventListener('keydown', event => { if (event.key === 'Escape') reset(); }, options);
    window.addEventListener('blur', reset, options);
    document.addEventListener('visibilitychange', () => { if (document.hidden) reset(); }, options);
    document.addEventListener('dragover', event => {
      if (!event.dataTransfer?.types.includes('Files')) return;
      const root = event.target instanceof Element ? event.target.closest<HTMLElement>('[data-composition="true"]') : null;
      if (!root) { reset(); return; }
      event.preventDefault(); event.dataTransfer.dropEffect = 'copy';
      destination = locate(root, event.clientX, event.clientY, 'image');
    }, options);
    document.addEventListener('drop', event => {
      if (!event.dataTransfer?.types.includes('Files')) return;
      event.preventDefault();
      const root = event.target instanceof Element ? event.target.closest<HTMLElement>('[data-composition="true"]') : null;
      if (!root) { reset(); return; }
      const d = locate(root, event.clientX, event.clientY, 'image');
      const files = event.dataTransfer.files;
      if (files.length !== 1 || !['image/png', 'image/jpeg', 'image/webp'].includes(files[0].type) || files[0].size > 8_000_000)
        send({ action: 'error', message: 'Drop one PNG, JPEG or WebP image, up to 8 MB.' });
      else if (d) send({ action: 'file', sectionId: root.id, destination: d, mobile: innerWidth <= 640, file: files[0] });
      reset();
    }, options);
    document.addEventListener('dragend', reset, options);
    document.addEventListener('dragleave', event => { if (!event.relatedTarget) reset(); }, options);
    cleanup = () => { reset(); abort.abort(); handles.forEach(handle => handle.remove()); emptyParts.forEach(el => el.classList.remove('pg-compose-empty')); style.remove(); };
  }
  mount();
  window.addEventListener('message', event => {
    if (event.source !== parent || !event.data || event.data.type !== 'playground-update') return;
    config = event.data.config;
    mount();
  });
  window.addEventListener('pagehide', () => cleanup(), { once: true });
}
