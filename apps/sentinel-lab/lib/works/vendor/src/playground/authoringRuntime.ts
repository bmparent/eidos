/** Self-contained authoring interaction, excluded from exported script.js. */
export function authoringRuntime(initial: {
    bridge: boolean;
    editing: boolean;
    channel?: string;
    compositionStamp?: string;
    revision?: number;
}) {
    if (!initial.bridge)
        return;
    let config = initial, cleanup = () => { };
    let selected = '';
    window.addEventListener('message', e => {
        if(e.source===parent && e.data?.channel===initial.channel && e.data.type==='playground-cancel' && typeof e.data.scale==='number' && e.data.scale>0 && e.data.scale<=1)
            document.documentElement.style.setProperty('--pg-chrome-unit', (1/e.data.scale)+'px');
    });
    function mount() {
        cleanup();
        if (!config.editing || !config.compositionStamp)
            return;
        const abort = new AbortController(), options = { signal: abort.signal };
        const roots = Array.from(document.querySelectorAll<HTMLElement>('[data-authoring]'));
        if (!roots.length)
            return;
        const all = roots.flatMap(root => Array.from(root.querySelectorAll<HTMLElement>('[data-node]')));
        const style = document.createElement('style');
        style.textContent = `.pg-node-selected{outline:2px solid #bce886;outline-offset:3px}.pg-node-toolbar{position:fixed;z-index:11000;display:flex;gap:3px;flex-wrap:wrap;max-width:calc(100vw - 12px);padding:4px;background:#15242c;color:white;border:1px solid #bce886;border-radius:8px;box-shadow:0 3px 16px #0004}.pg-node-toolbar button,.pg-node-resize{font:600 calc(12 * var(--pg-chrome-unit,1px)) Arial,sans-serif;min-height:calc(44 * var(--pg-chrome-unit,1px));min-width:calc(44 * var(--pg-chrome-unit,1px));border:0;border-radius:5px;background:#243c47;color:white;cursor:pointer}.pg-node-toolbar [data-drag],.pg-node-resize{touch-action:none;cursor:grab}.pg-node-resize{position:absolute;right:0;bottom:0;z-index:10;cursor:nwse-resize}.pg-node-ghost{position:fixed!important;left:0!important;top:0!important;inset:auto!important;z-index:10998!important;pointer-events:none!important;opacity:.8!important;margin:0!important;outline:2px solid #bce886;overflow:hidden}.pg-node-ghost *{pointer-events:none!important}.pg-node-placeholder{opacity:.25!important}.pg-node-destination{position:fixed;z-index:10999;pointer-events:none;border:2px dashed #bce886;background:#142b38dc;color:white;font:600 12px Arial;padding:6px;border-radius:5px;box-sizing:border-box}.pg-node [contenteditable]{outline:2px solid #bce886;user-select:text;cursor:text}.pg-node-resize[data-split]{right:calc(50% - 22px);top:0;bottom:auto;cursor:col-resize}`;
        document.head.append(style);
        let toolbar: HTMLElement | null = null, resize: HTMLButtonElement | null = null, split: HTMLButtonElement | null = null, marker: HTMLElement | null = null, frame = 0, mode: 'move' | 'crop' = 'move';
        type Target = {
            parent: HTMLElement;
            before: HTMLElement | null;
            label: string;
            rect: DOMRect;
        };
        type Gesture = {
            kind: 'move' | 'resize' | 'crop' | 'split';
            node: HTMLElement;
            handle: HTMLButtonElement;
            pointer: number;
            x: number;
            y: number;
            sx: number;
            sy: number;
            rect: DOMRect;
            parentRect: DOMRect;
            originalStyle: string | null;
            ghost: HTMLElement | null;
            stamp: string;
            active: boolean;
            patch: Record<string, unknown>;
            target: Target | null;
            time: number;
            lastSample: number;
            sample: number;
            dirty: boolean;
            geometry: Map<HTMLElement, DOMRect>;
            extra: {
                el: HTMLElement;
                style: string | null;
            }[];
        };
        let gesture: Gesture | null = null, textEdit: {
            el: HTMLElement;
            original: string;
            node: HTMLElement;
            stamp: string;
        } | null = null;
        function send(action: Record<string, unknown>, stamp = config.compositionStamp) { parent.postMessage({ type: 'playground-block', channel: config.channel, stamp, ...action }, '*'); }
        function clearChrome() { toolbar?.remove(); resize?.remove(); split?.remove(); toolbar = null; resize = null; split = null; }
        function cancel() { const old = gesture; gesture = null; cancelAnimationFrame(frame); frame = 0; marker?.remove(); marker = null; delete document.body.dataset.gesture; if (old) {
            old.ghost?.remove();
            old.node.classList.remove('pg-node-placeholder');
            if (old.originalStyle === null)
                old.node.removeAttribute('style');
            else
                old.node.setAttribute('style', old.originalStyle);
            for (const v of old.extra) {
                if (v.style === null)
                    v.el.removeAttribute('style');
                else
                    v.el.setAttribute('style', v.style);
            }
            if (old.handle.hasPointerCapture(old.pointer))
                old.handle.releasePointerCapture(old.pointer);
        } }
        function finishText(commit: boolean) { const old = textEdit; if (!old)
            return; textEdit = null; const text = old.el.textContent || ''; old.el.removeAttribute('contenteditable'); old.el.textContent = old.original; if (commit && text !== old.original)
            send({ action: 'command', operation: { type: 'patch', id: old.node.id, patch: { text } } }, old.stamp); }
        function position() { const el = document.getElementById(selected); if (!el || !toolbar)
            return; const r = el.getBoundingClientRect(); toolbar.style.left = Math.max(6, Math.min(innerWidth - toolbar.offsetWidth - 6, r.left)) + 'px'; toolbar.style.top = Math.max(6, Math.min(innerHeight - toolbar.offsetHeight - 6, r.top - toolbar.offsetHeight - 5)) + 'px'; }
        function button(text: string, fn: () => void) { const b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); fn(); }, options); toolbar!.append(b); return b; }
        function structureButtons(node: HTMLElement) {
            const section=node.closest<HTMLElement>('[data-section]');
            if(!section || section.id==='header' || section.id==='footer')return;
            const isCard=node.classList.contains('pg-card'), id=isCard?node.id:section.id;
            const prefix=isCard?'card':'section';
            for(const direction of [-1,1] as const) button('Move '+prefix+(direction===-1?' earlier':' later'),()=>send({action:'command',operation:{type:prefix+'-step',id,direction}}));
        }
        function selectStructure(node: HTMLElement) {
            cancel();finishText(true);clearChrome();all.forEach(n=>n.classList.remove('pg-node-selected'));
            selected=node.id;toolbar=document.createElement('div');toolbar.className='pg-node-toolbar';toolbar.setAttribute('role','toolbar');toolbar.setAttribute('aria-label','Structure actions');document.body.append(toolbar);
            structureButtons(node);position();
        }
        function select(node: HTMLElement, notify = true) {
            if (selected === node.id && toolbar)
                return;
            cancel();
            finishText(true);
            clearChrome();
            all.forEach(n => n.classList.toggle('pg-node-selected', n === node));
            selected = node.id;
            mode = 'move';
            toolbar = document.createElement('div');
            toolbar.className = 'pg-node-toolbar';
            toolbar.setAttribute('role', 'toolbar');
            toolbar.setAttribute('aria-label', node.dataset.nodeName || 'Selected block');
            document.body.append(toolbar);
            const parent = node.parentElement?.closest<HTMLElement>('[data-node-container]');
            if(!parent)structureButtons(node);
            if (parent) {
                const move = button('Move', () => send({ action: 'inspect', id: node.id, sectionId: node.closest('[data-authoring]')!.id }));
                move.dataset.drag = 'move';
                move.setAttribute('aria-label', 'Drag selected block');
                move.addEventListener('pointerdown', e => start(e, node, 'move'), options);
                button('Duplicate', () => send({ action: 'command', operation: { type: 'duplicate', id: node.id } }));
                button('Remove', () => send({ action: 'command', operation: { type: 'remove', id: node.id } }));
            }
            if (node.dataset.nodeType === 'image')
                button('Crop', () => { mode = mode === 'crop' ? 'move' : 'crop'; node.style.touchAction = mode === 'crop' ? 'none' : ''; toolbar!.dataset.mode = mode; const hint = toolbar!.querySelector('[data-crop-hint]'); hint?.remove(); if (mode === 'crop') {
                    const h = document.createElement('span');
                    h.dataset.cropHint = 'true';
                    h.textContent = 'Drag photo to reposition';
                    toolbar!.append(h);
                } });
            const editable = node.querySelector<HTMLElement>(':scope > [data-node-text]');
            if (editable)
                button('Edit text', () => { finishText(true); textEdit = { el: editable, original: editable.textContent || '', node, stamp: config.compositionStamp! }; editable.contentEditable = 'true'; editable.focus(); const range = document.createRange(); range.selectNodeContents(editable); const selection = getSelection(); selection?.removeAllRanges(); selection?.addRange(range); });
            button('Controls', () => send({ action: 'inspect', id: node.id, sectionId: node.closest('[data-authoring]')!.id }));
            resize = document.createElement('button');
            resize.type = 'button';
            resize.className = 'pg-node-resize';
            resize.textContent = '↘';
            resize.setAttribute('aria-label', node.dataset.nodeType === 'image' ? 'Resize image frame' : 'Resize block width');
            node.append(resize);
            resize.addEventListener('pointerdown', e => start(e, node, 'resize'), options);
            if (innerWidth > 640 && node.dataset.nodeType === 'columns' && node.querySelectorAll(':scope > [data-node]').length === 2) {
                split = document.createElement('button');
                split.type = 'button';
                split.className = 'pg-node-resize';
                split.dataset.split = 'true';
                split.textContent = '↔';
                split.setAttribute('aria-label', 'Adjust column split');
                node.append(split);
                split.addEventListener('pointerdown', e => start(e, node, 'split'), options);
            }
            position();
            if (notify)
                send({ action: 'select', id: node.id, sectionId: node.closest('[data-authoring]')!.id });
        }
        function start(event: PointerEvent, node: HTMLElement, kind: Gesture['kind']) { if (!event.isPrimary || event.button !== 0)
            return; event.preventDefault(); event.stopPropagation(); cancel(); finishText(true); const rect = node.getBoundingClientRect(); gesture = { kind, node, handle: event.currentTarget as HTMLButtonElement, pointer: event.pointerId, x: event.clientX, y: event.clientY, sx: event.clientX, sy: event.clientY, rect, parentRect: node.parentElement!.getBoundingClientRect(), originalStyle: node.getAttribute('style'), ghost: null, stamp: config.compositionStamp!, active: false, patch: {}, target: null, time: 0, lastSample: performance.now(), sample: 0, dirty: true, geometry: new Map(), extra: [] }; gesture.handle.setPointerCapture(event.pointerId); }
        function destination(g: Gesture): Target | null {
            if (g.dirty) {
                g.geometry = new Map(all.map(el => [el, el.getBoundingClientRect()]));
                g.dirty = false;
            }
            if (g.x < 0 || g.x > innerWidth || g.y < 0 || g.y > innerHeight)
                return null;
            const hit = document.elementFromPoint(g.x, g.y);
            const parent = hit?.closest<HTMLElement>('[data-node-container]');
            if (!parent || parent === g.node || g.node.contains(parent))
                return null;
            const rect = g.geometry.get(parent)!;
            const children = Array.from(parent.querySelectorAll<HTMLElement>(':scope > [data-node]')).filter(n => n !== g.node);
            const horizontal = getComputedStyle(parent).flexDirection === 'row';
            const before = children.find(n => { const r = g.geometry.get(n)!; return horizontal ? g.x < r.left + r.width / 2 : g.y < r.top + r.height / 2; }) || null;
            const r = before ? g.geometry.get(before)! : rect;
            return { parent, before, label: before ? 'Before ' + before.dataset.nodeName : 'Inside ' + (parent.dataset.nodeName || 'this stack'), rect: new DOMRect(horizontal ? (before ? r.left : rect.right - 20) : rect.left, horizontal ? rect.top : (before ? r.top : rect.bottom - 22), horizontal ? 24 : rect.width, horizontal ? rect.height : 24) };
        }
        const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));
        function sample(g: Gesture) {
            const dx = g.x - g.sx, dy = g.y - g.sy, mobile = innerWidth <= 640;
            if (g.kind === 'move') {
                const next = destination(g);
                if (!g.target || !next || next.parent !== g.target.parent || next.before !== g.target.before) {
                    if (!g.target || !next || Math.hypot(g.x - (g.target.rect.x + g.target.rect.width / 2), g.y - (g.target.rect.y + g.target.rect.height / 2)) > 10)
                        g.target = next;
                }
                if (!marker) {
                    marker = document.createElement('div');
                    marker.className = 'pg-node-destination';
                    document.body.append(marker);
                }
                const t = g.target;
                marker.hidden = !t;
                if (t) {
                    Object.assign(marker.style, { left: t.rect.x + 'px', top: t.rect.y + 'px', width: Math.max(40, t.rect.width) + 'px', minHeight: '24px' });
                    marker.textContent = t.label;
                }
                if (g.ghost)
                    g.ghost.style.transform = `translate3d(${g.x - (g.sx - g.rect.left)}px,${g.y - (g.sy - g.rect.top)}px,0)`;
            }
            else if (g.kind === 'resize') {
                let width = clamp((g.rect.width + dx) / g.parentRect.width * 100, 10, 100);
                const height = clamp(g.node.dataset.nodeLock === 'true' ? width / 100 * g.parentRect.width * g.rect.height / g.rect.width : g.rect.height + dy, 64, 960);
                if(g.node.dataset.nodeLock==='true' && g.node.dataset.nodeType==='image')width=clamp(height*g.rect.width/g.rect.height/g.parentRect.width*100,10,100);
                g.patch = { width, ...(g.node.dataset.nodeType === 'image' ? { height } : {}) };
                g.node.style.width = width + '%';
                g.node.style.flex = '0 0 auto';
                if (g.node.dataset.nodeType === 'image')
                    g.node.style.height = height + 'px';
                if (mobile)
                    g.patch = { mobile: { ...JSON.parse(g.node.dataset.nodeMobile || '{}'), ...g.patch } };
            }
            else if (g.kind === 'split') {
                const split = clamp((g.x - g.rect.left) / g.rect.width * 100, 20, 80);
                g.patch = { split };
                const children = Array.from(g.node.querySelectorAll<HTMLElement>(':scope > [data-node]'));
                children[0].style.flex = split + ' 1 0';
                children[1].style.flex = (100 - split) + ' 1 0';
            }
            else {
                const m = JSON.parse(g.node.dataset.nodeMedia || '{}'), x = clamp((mobile ? m.mobileX ?? m.x : m.x) - dx / g.rect.width * 100, 0, 100), y = clamp((mobile ? m.mobileY ?? m.y : m.y) - dy / g.rect.height * 100, 0, 100);
                g.patch = { media: { ...m, ...(mobile ? { mobileX: x, mobileY: y } : { x, y }) } };
                const img = g.node.querySelector('img');
                if (img)
                    img.style.objectPosition = `${x}% ${y}%`;
            }
        }
        function paint(t: number) {
            frame = 0;
            const g = gesture;
            if (!g?.active)
                return;
            const dt = Math.min(32, t - (g.time || t - 16));
            g.time = t;
            if (g.kind === 'move') {
                for (let el: HTMLElement | null = g.node.parentElement; el; el = el.parentElement) {
                    const page = el === document.scrollingElement;
                    if (!page && !/(auto|scroll)/.test(getComputedStyle(el).overflowY))
                        continue;
                    const r = page ? { top: 0, bottom: innerHeight } : el.getBoundingClientRect(), speed = g.y < r.top + 40 ? -.5 : g.y > r.bottom - 40 ? .5 : 0;
                    const old = el.scrollTop;
                    el.scrollTop += speed * dt;
                    if (el.scrollTop !== old) {
                        g.dirty = true;
                        break;
                    }
                }
            }
            sample(g);
            position();
            if (g.sample !== g.lastSample) {
                g.sample = g.lastSample;
                document.dispatchEvent(new CustomEvent('playground-gesture-paint', { detail: { sample: g.lastSample, paint: performance.now() } }));
            }
            frame = requestAnimationFrame(paint);
        }
        document.addEventListener('click', e=>{
            const target=e.target as Element;
            if(target.closest('[data-node],.pg-node-toolbar'))return;
            const node=target.closest<HTMLElement>('.pg-card[id],[data-section]');
            if(node && node.id!=='header' && node.id!=='footer')selectStructure(node);
        },options);
        for (const node of all) {
            node.addEventListener('click', event => { event.preventDefault(); event.stopPropagation(); if (textEdit?.node === node)
                return; select(node); }, options);
            node.addEventListener('pointerdown', event => { if (selected === node.id && mode === 'crop' && node.dataset.nodeType === 'image' && !((event.target as Element).closest('button'))) {
                start(event, node, 'crop');
            } }, options);
        }
        document.addEventListener('pointermove', e => {
            const g = gesture;
            if (!g || e.pointerId !== g.pointer)
                return;
            g.x = e.clientX;
            g.y = e.clientY;
            g.lastSample = performance.now();
            if (!g.active && Math.hypot(g.x - g.sx, g.y - g.sy) < 6)
                return;
            e.preventDefault();
            if (!g.active) {
                g.active = true;
                document.body.dataset.gesture = g.kind;
                if (g.kind === 'move') {
                    const ghost = g.node.cloneNode(true) as HTMLElement;
                    const originals = [g.node, ...g.node.querySelectorAll<HTMLElement>('*')], copies = [ghost, ...ghost.querySelectorAll<HTMLElement>('*')];
                    for (let i = 0; i < copies.length; i++) {
                        const css = getComputedStyle(originals[i]);
                        Object.assign(copies[i].style, { font: css.font, color: css.color, objectFit: css.objectFit, objectPosition: css.objectPosition, transform: css.transform });
                    }
                    ghost.querySelectorAll('button').forEach(n => n.remove());
                    ghost.querySelectorAll('[id],[data-node]').forEach(n => { n.removeAttribute('id'); n.removeAttribute('data-node'); });
                    ghost.removeAttribute('id');
                    ghost.removeAttribute('data-node');
                    ghost.className += ' pg-node-ghost';
                    ghost.inert = true;
                    ghost.setAttribute('aria-hidden', 'true');
                    Object.assign(ghost.style, { width: g.rect.width + 'px', height: g.rect.height + 'px', font: getComputedStyle(g.node).font, color: getComputedStyle(g.node).color });
                    document.body.append(ghost);
                    g.ghost = ghost;
                    g.node.classList.add('pg-node-placeholder');
                }
                else if (g.kind === 'split')
                    g.extra = Array.from(g.node.querySelectorAll<HTMLElement>(':scope > [data-node]')).map(el => ({ el, style: el.getAttribute('style') }));
                else if (g.kind === 'crop') {
                    const img = g.node.querySelector('img');
                    if (img)
                        g.extra = [{ el: img, style: img.getAttribute('style') }];
                }
                frame = requestAnimationFrame(paint);
            }
        }, { ...options, passive: false });
        document.addEventListener('pointerup', e => { const g = gesture; if (!g || g.pointer !== e.pointerId)
            return; g.x = e.clientX; g.y = e.clientY; g.dirty = true; if (g.active)
            sample(g); const target = g.kind === 'move' ? destination(g) : null; const operation = g.kind === 'move' ? (target ? { type: 'move', id: g.node.id, parent: target.parent.id, before: target.before?.id || null } : null) : { type: 'patch', id: g.node.id, patch: g.patch }; const commit = g.active && operation && g.x>=0 && g.x<=innerWidth && g.y>=0 && g.y<=innerHeight && (g.kind==='move'||Math.hypot(g.x-g.sx,g.y-g.sy)>.5); cancel(); if (commit)
            send({ action: 'command', operation }, g.stamp); }, options);
        window.visualViewport?.addEventListener('resize',cancel,options);
        document.addEventListener('pointercancel', cancel, options);
        document.addEventListener('lostpointercapture', () => { if (gesture)
            cancel(); }, options);
        window.addEventListener('blur', () => { cancel(); finishText(false); }, options);
        window.addEventListener('resize', () => { cancel(); finishText(false); position(); }, options);
        document.addEventListener('visibilitychange', () => { if (document.hidden) {
            cancel();
            finishText(false);
        } }, options);
        document.addEventListener('scroll', () => { if (gesture)
            gesture.dirty = true; position(); }, { ...options, passive: true, capture: true });
        document.addEventListener('keydown', e => { if (e.key === 'Escape') {
            cancel();
            finishText(false);
        } if (textEdit && e.key === 'Enter' && (textEdit.node.dataset.nodeType !== 'text' || e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            finishText(true);
        } }, options);
        document.addEventListener('focusout', e => { if (textEdit?.el === e.target)
            finishText(true); }, options);
        document.addEventListener('paste', e => { if (!textEdit)
            return; e.preventDefault(); const s = getSelection(); if (!s?.rangeCount)
            return; const r = s.getRangeAt(0); r.deleteContents(); const n = document.createTextNode(e.clipboardData?.getData('text/plain') || ''); r.insertNode(n); r.setStartAfter(n); r.collapse(true); s.removeAllRanges(); s.addRange(r); }, options);
        window.addEventListener('playground-before-update', () => { finishText(false); cleanup(); }, options);
        window.addEventListener('message', e => { if (e.source === parent && e.data?.channel === initial.channel && e.data.type === 'playground-cancel') {
            cancel();
            finishText(false);
            return;
        } if (e.source === parent && e.data?.channel === initial.channel && e.data.type === 'playground-block-selection') {
            const n = all.find(n => n.id === e.data.id);
            if (n)
                select(n, false);
        } }, options);
        cleanup = () => { cancel(); finishText(false); clearChrome(); abort.abort(); all.forEach(n => { n.classList.remove('pg-node-selected'); n.style.removeProperty('touch-action'); }); style.remove(); };
        const current = all.find(n => n.id === selected);
        if (current)
            select(current, false);
    }
    mount();
    window.addEventListener('message', e => { if (e.source !== parent || e.data?.channel !== initial.channel || e.data.type !== 'playground-update')
        return; const next = e.data.config; if (!next || next.channel !== initial.channel || !Number.isSafeInteger(next.revision) || next.revision <= (config.revision || 0))
        return; config = next; mount(); });
    window.addEventListener('pagehide', () => cleanup(), { once: true });
}
