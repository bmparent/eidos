export class PatternMemory {
  constructor({ capacity = 64, sampleStride = 2 } = {}) { this.capacity=capacity; this.sampleStride=sampleStride; this.ring=[]; }
  fingerprint(alive,width,height){ const bits=[]; for(let y=0;y<height;y+=this.sampleStride) for(let x=0;x<width;x+=this.sampleStride) bits.push(alive[y*width+x]); return Uint8Array.from(bits); }
  hamming(a,b){ const n=Math.min(a.length,b.length); let d=0; for(let i=0;i<n;i++) d += a[i]===b[i]?0:1; return d/Math.max(1,n); }
  novelty(fp){ if(!this.ring.length) return 1; let best=1; for(const item of this.ring) best=Math.min(best,this.hamming(fp,item)); return best; }
  remember(fp){ this.ring.push(fp); if(this.ring.length>this.capacity) this.ring.shift(); }
  classifyPatch(alive,width,height){
    // approximate heuristic only
    let live=0; for(let i=0;i<alive.length;i++) live+=alive[i]; if(live===0) return 'unknown';
    if(live===3) return 'blinker'; if(live===4) return 'block'; if(live===6) return 'beacon'; if(live===5) return 'glider';
    return 'unknown';
  }
}
