/** Editor-only pointer presentation. Never included in standalone exports. */
export function compositionRuntime(initial: { editing: boolean; bridge: boolean; compositionStamp?: string; channel?: string; revision?: number }) {
  if (!initial.bridge) return;
  let config = initial;
  let cleanup = () => {};
  let selected = '';
  const labels: Record<string, string> = { title: 'Heading', description: 'Description', cta: 'Button', image: 'Image' };
  function mount() {
    cleanup();
    if (!config.editing || !config.compositionStamp) return;
    const abort = new AbortController(), options = { signal: abort.signal };
    const roots = Array.from(document.querySelectorAll<HTMLElement>('[data-composition="true"]'));
    const style = document.createElement('style');
    style.dataset.compositionEditor = 'true';
    style.textContent = `
      .pg-part.pg-part-selected{outline:2px solid #bce886;outline-offset:3px}
      .pg-compose-handle{position:absolute;top:0;right:0;z-index:5;background:#15242c;color:#f5f8f4;border:1px solid #bce886;border-radius:7px;font:600 12px/1.2 Arial,sans-serif;min-width:44px;min-height:44px;padding:7px;touch-action:none;cursor:grab;pointer-events:auto;box-shadow:0 2px 6px #0004}
      .pg-compose-handle:focus-visible{outline:3px solid #fff;outline-offset:2px}
      .pg-composed>.pg-compose-handle{top:8px;right:8px}
      .pg-compose-hud{position:fixed;z-index:10002;bottom:12px;left:50%;transform:translateX(-50%);max-width:calc(100vw - 24px);width:max-content;background:#15242c;color:#fff;border:1px solid #bce886;border-radius:10px;font:600 13px/1.5 Arial,sans-serif;padding:8px 12px;pointer-events:none}
      .pg-compose-target{position:fixed;z-index:10001;border:2px dashed #bce886;background:#15242cdd;color:white;border-radius:6px;font:600 12px/1.3 Arial,sans-serif;padding:6px;pointer-events:none;box-sizing:border-box}
      .pg-compose-target[data-active=true]{background:#bce886;color:#15242c;border-style:solid}
      .pg-compose-ghost{position:fixed!important;inset:auto!important;z-index:10000!important;pointer-events:none!important;opacity:.85!important;margin:0!important;max-width:none!important;box-sizing:border-box;will-change:transform;outline:2px solid #bce886;overflow:hidden;isolation:isolate}
      .pg-compose-ghost *{pointer-events:none!important}.pg-compose-placeholder{opacity:.25!important;outline:2px dashed #bce886}
      .pg-compose-dragging{cursor:grabbing!important;user-select:none}
    `;
    document.head.append(style);
    type Destination = { placement: string } | { before: string | null };
    type Target = { key: string; label: string; destination: Destination; rect: { left:number;top:number;width:number;height:number }; node?: HTMLElement };
    type Gesture = { pointer:number;x:number;y:number;startX:number;startY:number;grabX:number;grabY:number;part:string;root:HTMLElement;el:HTMLElement;handle:HTMLButtonElement;stamp:string;moved:boolean;ghost:HTMLElement|null;targets:Target[];rootRect:DOMRect;scrollParents:HTMLElement[];dirty:boolean;lastTime:number;lastSample:number;paintSample:number;switchX:number;switchY:number };
    const handles: HTMLButtonElement[] = [];
    let gesture: Gesture | null = null, destination: Target | null = null, hud: HTMLElement | null = null, frame = 0;
    let suppressClick = false;
    function send(action: Record<string, unknown>, stamp = config.compositionStamp) {
      parent.postMessage({ type:'playground-composition', channel:config.channel, stamp, ...action }, '*');
    }
    function hint(text:string) {
      if (!hud) { hud=document.createElement('div');hud.className='pg-compose-hud';hud.setAttribute('role','status');document.body.append(hud); }
      if(hud.textContent!==text)hud.textContent=text;
    }
    function reset() {
      const old=gesture;gesture=null;destination=null;cancelAnimationFrame(frame);frame=0;
      old?.ghost?.remove();old?.el.classList.remove('pg-compose-placeholder');old?.targets.forEach(t=>t.node?.remove());
      hud?.remove();hud=null;document.body.classList.remove('pg-compose-dragging');delete document.body.dataset.gesture;
      if(old?.handle.hasPointerCapture(old.pointer))old.handle.releasePointerCapture(old.pointer);
    }
    function measure(g:Gesture) {
      g.targets.forEach(t=>t.node?.remove());
      const r=g.root.getBoundingClientRect();g.rootRect=r;
      const parts=Array.from(g.root.querySelectorAll<HTMLElement>('[data-composition-part]')).filter(el=>el.dataset.compositionPart!==g.part&&el.dataset.compositionPart!=='image');
      const rects=parts.map(el=>({el,r:el.getBoundingClientRect()}));
      const left=rects.length?Math.min(...rects.map(v=>v.r.left)):r.left+24;
      const right=rects.length?Math.max(...rects.map(v=>v.r.right)):r.right-24;
      const top=rects.length?Math.min(...rects.map(v=>v.r.top)):r.top+24;
      const bottom=rects.length?Math.max(...rects.map(v=>v.r.bottom)):r.bottom-24;
      const ts:Target[]=rects.map(({el,r:pr})=>({key:'before-'+el.dataset.compositionPart,label:'Before '+labels[el.dataset.compositionPart!].toLowerCase(),destination:{before:el.dataset.compositionPart!},rect:{left:pr.left,top:pr.top-12,width:pr.width,height:24}}));
      ts.push({key:'after',label:'After content',destination:{before:null},rect:{left,top:bottom,width:Math.max(80,right-left),height:24}});
      if(g.part==='image'&&innerWidth>640) {
        ts.push({key:'left',label:'Left column',destination:{placement:'left'},rect:{left:r.left+4,top:Math.max(r.top+36,top),width:88,height:44}});
        ts.push({key:'right',label:'Right column',destination:{placement:'right'},rect:{left:r.right-92,top:Math.max(r.top+36,top),width:88,height:44}});
      }
      g.targets=ts;g.dirty=false;
      // Geometry reads finish before presentation writes.
      for(const t of ts){const n=document.createElement('div');n.className='pg-compose-target';n.textContent=t.label;n.dataset.target=t.key;Object.assign(n.style,{left:t.rect.left+'px',top:t.rect.top+'px',width:t.rect.width+'px',minHeight:t.rect.height+'px'});document.body.append(n);t.node=n;}
      if(destination)destination=ts.find(t=>t.key===destination?.key)||null;
    }
    function locate(g:Gesture,final=false):Target|null {
      if(g.dirty)measure(g);
      const r=g.rootRect,x=g.x,y=g.y;
      if(x<Math.max(0,r.left)||x>Math.min(innerWidth,r.right)||y<Math.max(0,r.top)||y>Math.min(innerHeight,r.bottom))return null;
      const distance=(t:Target)=>Math.hypot(Math.max(t.rect.left-x,0,x-t.rect.left-t.rect.width),Math.max(t.rect.top-y,0,y-t.rect.top-t.rect.height));
      const next=g.targets.reduce<Target|null>((best,t)=>!best||distance(t)<distance(best)?t:best,null);
      if(!final&&destination&&next?.key!==destination.key&&distance(next!)+8>=distance(destination))return destination;
      return next;
    }
    function paint(time:number) {
      frame=0;const g=gesture;if(!g?.moved)return;
      const dt=Math.min(32,time-(g.lastTime||time-16));g.lastTime=time;
      // Scroll the closest scrollable container first, then its ancestors. Keep running for a stationary pointer.
      let scrolled=false;
      for(const scroller of g.scrollParents){const isPage=scroller===document.scrollingElement;const r=isPage?{top:0,bottom:innerHeight}:scroller.getBoundingClientRect();const edge=40;const speed=g.y<r.top+edge?-Math.min(1,(r.top+edge-g.y)/edge):g.y>r.bottom-edge?Math.min(1,(g.y-r.bottom+edge)/edge):0;if(!speed)continue;const previous=scroller.scrollTop;scroller.scrollTop+=speed*dt*.55;if(scroller.scrollTop!==previous){scrolled=true;break;}}
      if(scrolled)g.dirty=true;
      const next=locate(g);
      if(destination?.key!==next?.key){destination=next;g.switchX=g.x;g.switchY=g.y;}
      for(const t of g.targets)t.node!.dataset.active=String(t.key===destination?.key);
      hint(destination?destination.label+' · release to move · Esc cancels':'Outside supported targets · release to cancel');
      if(g.ghost)g.ghost.style.transform=`translate3d(${g.x-g.grabX}px,${g.y-g.grabY}px,0)`;
      if(g.paintSample!==g.lastSample){g.paintSample=g.lastSample;document.dispatchEvent(new CustomEvent('playground-gesture-paint',{detail:{sample:g.lastSample,paint:performance.now(),x:g.x,y:g.y}}));}
      frame=requestAnimationFrame(paint);
    }
    function activate(g:Gesture) {
      g.moved=true;document.body.dataset.gesture='drag';document.body.classList.add('pg-compose-dragging');
      const rect=g.el.getBoundingClientRect(),computed=getComputedStyle(g.el);
      const ghost=g.el.cloneNode(true) as HTMLElement;
      ghost.removeAttribute('id');ghost.removeAttribute('data-composition-part');ghost.querySelectorAll('[id]').forEach(e=>e.removeAttribute('id'));ghost.querySelectorAll('.pg-compose-handle').forEach(e=>e.remove());
      ghost.className+=' pg-compose-ghost';ghost.classList.remove('pg-part-selected');ghost.setAttribute('aria-hidden','true');ghost.inert=true;
      Object.assign(ghost.style,{left:'0',top:'0',width:rect.width+'px',height:rect.height+'px',font:computed.font,color:computed.color,background:computed.backgroundColor});
      document.body.append(ghost);g.ghost=ghost;g.el.classList.add('pg-compose-placeholder');g.dirty=true;
      frame=requestAnimationFrame(paint);
    }
    function select(root:HTMLElement,el:HTMLElement,part:string) {
      root.querySelectorAll('.pg-part-selected').forEach(e=>e.classList.remove('pg-part-selected'));el.classList.add('pg-part-selected');selected=root.id+':'+part;
      send({action:'select',sectionId:root.id,part});
    }
    for(const root of roots)for(const el of root.querySelectorAll<HTMLElement>('[data-composition-part]')) {
      const part=el.dataset.compositionPart!;
      if(selected===root.id+':'+part)el.classList.add('pg-part-selected');
      const handle=document.createElement('button');handle.type='button';handle.className='pg-compose-handle';handle.textContent='↕ '+labels[part];handle.setAttribute('aria-label',`Move ${labels[part].toLowerCase()}. Drag, or use Layout controls.`);
      const background=part==='image'&&getComputedStyle(el).position==='absolute';
      if(background){root.append(handle);handle.textContent='↕ Background image';}else el.append(handle);
      handles.push(handle);
      el.addEventListener('click',event=>{if(suppressClick){suppressClick=false;event.preventDefault();event.stopPropagation();return;}event.preventDefault();event.stopPropagation();select(root,el,part);},options);
      handle.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();if(suppressClick){suppressClick=false;return;}select(root,el,part);},options);
      handle.addEventListener('keydown',event=>{
        if(!['ArrowUp','ArrowDown'].includes(event.key))return;event.preventDefault();event.stopPropagation();
        const parts=Array.from(root.querySelectorAll<HTMLElement>('[data-composition-part]')).map(e=>e.dataset.compositionPart!);const index=parts.indexOf(part);if(event.key==='ArrowUp'&&index>0)send({action:'move',sectionId:root.id,part,destination:{before:parts[index-1]},mobile:innerWidth<=640});if(event.key==='ArrowDown'&&index<parts.length-1)send({action:'move',sectionId:root.id,part,destination:{before:parts[index+2]||null},mobile:innerWidth<=640});
      },options);
      handle.addEventListener('pointerdown',event=>{
        if(!event.isPrimary||event.button!==0)return;event.preventDefault();event.stopPropagation();reset();suppressClick=false;
        const r=el.getBoundingClientRect();const scrollParents:HTMLElement[]=[];
        for(let node:HTMLElement|null=el.parentElement;node;node=node.parentElement)if(node.scrollHeight>node.clientHeight&&/(auto|scroll)/.test(getComputedStyle(node).overflowY))scrollParents.push(node);
        if(document.scrollingElement)scrollParents.push(document.scrollingElement as HTMLElement);
        gesture={pointer:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,grabX:event.clientX-r.left,grabY:event.clientY-r.top,part,root,el,handle,stamp:config.compositionStamp!,moved:false,ghost:null,targets:[],rootRect:root.getBoundingClientRect(),scrollParents,dirty:true,lastTime:0,lastSample:performance.now(),paintSample:0,switchX:event.clientX,switchY:event.clientY};
        handle.focus({preventScroll:true});handle.setPointerCapture(event.pointerId);select(root,el,part);
      },options);
    }
    document.addEventListener('pointermove',event=>{
      const g=gesture;if(!g||g.pointer!==event.pointerId)return;g.x=event.clientX;g.y=event.clientY;g.lastSample=performance.now();
      if(!g.moved&&Math.hypot(g.x-g.startX,g.y-g.startY)<6)return;
      event.preventDefault();if(!g.moved)activate(g);
    },{...options,passive:false});
    document.addEventListener('pointerup',event=>{
      const g=gesture;if(!g||g.pointer!==event.pointerId)return;g.x=event.clientX;g.y=event.clientY;g.dirty=true;
      const d=g.moved?locate(g,true):null;suppressClick=g.moved;reset();
      if(d)send({action:'move',sectionId:g.root.id,part:g.part,destination:d.destination,mobile:innerWidth<=640},g.stamp);
    },options);
    document.addEventListener('pointercancel',reset,options);document.addEventListener('lostpointercapture',()=>{if(gesture)reset();},options);
    document.addEventListener('keydown',event=>{if(event.key==='Escape'){suppressClick=!!gesture;reset();}},options);
    window.addEventListener('blur',reset,options);window.addEventListener('resize',reset,options);window.visualViewport?.addEventListener('resize',reset,options);
    document.addEventListener('visibilitychange',()=>{if(document.hidden)reset();},options);
    document.addEventListener('scroll',()=>{if(gesture)gesture.dirty=true;},{...options,capture:true,passive:true});
    const observer=new ResizeObserver(()=>{if(gesture)reset();});roots.forEach(root=>observer.observe(root));
    // Native drops remain exclusively OS file uploads. Replacing the single legacy image is explicit.
    document.addEventListener('dragover',event=>{if(!event.dataTransfer?.types.includes('Files'))return;const image=event.target instanceof Element?event.target.closest('[data-composition-part="image"]'):null;if(image){event.preventDefault();event.dataTransfer.dropEffect='copy';hint('Replace this image · use Media / Brand for a background');}},options);
    document.addEventListener('drop',event=>{
      if(!event.dataTransfer?.types.includes('Files'))return;event.preventDefault();const image=event.target instanceof Element?event.target.closest('[data-composition-part="image"]'):null;const root=image?.closest<HTMLElement>('[data-composition="true"]');
      const files=event.dataTransfer.files;if(root&&files.length===1&&['image/png','image/jpeg','image/webp'].includes(files[0].type)&&files[0].size<=8_000_000)send({action:'file',sectionId:root.id,destination:{placement:root.dataset.placement||'inline'},mobile:innerWidth<=640,file:files[0]});else if(root)send({action:'error'});reset();
    },options);
    document.addEventListener('dragend',reset,options);document.addEventListener('dragleave',event=>{if(!event.relatedTarget)reset();},options);
    window.addEventListener('message',e=>{if(e.source===parent&&e.data?.channel===initial.channel&&e.data.type==='playground-cancel')reset();},options);
    window.addEventListener('playground-before-update',()=>cleanup(),options);
    cleanup=()=>{reset();abort.abort();observer.disconnect();handles.forEach(h=>h.remove());style.remove();};
  }
  mount();
  window.addEventListener('message',event=>{
    if(event.source!==parent||!event.data||event.data.channel!==initial.channel||event.data.type!=='playground-update')return;
    const next=event.data.config;if(!next||next.channel!==initial.channel||!Number.isSafeInteger(next.revision)||next.revision<=(config.revision||0))return;
    config=next;mount();
  });
  window.addEventListener('pagehide',()=>cleanup(),{once:true});
}
