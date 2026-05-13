export class TelemetryRecorder {
  constructor(){ this.startedAt=new Date().toISOString(); this.rows=[]; this.events=[]; this.prevRegime='CALIBRATING'; this.largestMass=0; }
  record(row, organisms=[]){
    this.rows.push(row);
    if (row.regime!==this.prevRegime) this.events.push({generation:row.generation,type:'regime_change',regime:row.regime,severity:'low',metrics:pick(row),description:`Regime changed ${this.prevRegime} -> ${row.regime}`});
    if (row.regime==='VIOLET') this.events.push({generation:row.generation,type:'rare_geometry',regime:row.regime,severity:'medium',metrics:pick(row),description:'Rare structure emergence after instability.'});
    if (row.collapseRisk) this.events.push({generation:row.generation,type:'collapse_protection',regime:row.regime,severity:'high',metrics:pick(row),description:'RED collapse protection condition active.'});
    if (row.surprise>0.25) this.events.push({generation:row.generation,type:'surprise_spike',regime:row.regime,severity:'medium',metrics:pick(row),description:'Surprise exceeded alert threshold.'});
    const biggest=organisms.reduce((m,o)=>Math.max(m,o.mass),0); if(biggest>this.largestMass+25){ this.largestMass=biggest; this.events.push({generation:row.generation,type:'large_organism',regime:row.regime,severity:'medium',metrics:pick(row),description:'New large organism emerged.'}); }
    this.prevRegime=row.regime;
  }
  exportBundle(){
    const summary={ generations:this.rows.length, peakSurprise:Math.max(0,...this.rows.map(r=>r.surprise)), violetFrames:this.rows.filter(r=>r.regime==='VIOLET').length, events:this.events.length };
    const manifest={ version:'0.2', experiment:'eidos-life-living-stream-lab', startedAt:this.startedAt, exportedAt:new Date().toISOString() };
    return { manifest, summary, telemetry:this.rows, interestingEvents:this.events };
  }
}
function pick(r){ return {surprise:r.surprise,entropy:r.entropy,compressionRatio:r.compressionRatio,novelty:r.novelty}; }
