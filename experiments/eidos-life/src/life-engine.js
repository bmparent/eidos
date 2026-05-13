export const DEFAULT_RULE = { birth: [3], survive: [2, 3], mutation: 0.0002, reseed: false };

export class LifeEngine {
  constructor({ width = 72, height = 72 } = {}) {
    this.width = width; this.height = height; this.size = width * height;
    this.alive = new Uint8Array(this.size);
    this.previousAlive = new Uint8Array(this.size);
    this.nextAlive = new Uint8Array(this.size);
    this.age = new Uint16Array(this.size);
    this.energy = new Float32Array(this.size).fill(0.65);
    this.species = new Uint8Array(this.size);
    this.memory = new Float32Array(this.size);
    this.stress = new Float32Array(this.size);
    this.signalField = new Float32Array(this.size);
    this.anomalyField = new Float32Array(this.size);
    this.memoryField = new Float32Array(this.size);
    this.generation = 0;
  }
  idx(x,y){ return ((y + this.height) % this.height) * this.width + ((x + this.width) % this.width); }
  countNeighbors(x,y){ let n=0; for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){ if(!dx&&!dy) continue; n += this.alive[this.idx(x+dx,y+dy)]; } return n; }
  seed(points){ this.clear(); for (const [x,y,s=1] of points){ const i=this.idx(x,y); this.alive[i]=1; this.energy[i]=Math.max(this.energy[i],0.4); this.species[i]=s; } }
  clear(){ this.alive.fill(0); this.previousAlive.fill(0); this.nextAlive.fill(0); this.age.fill(0); this.memory.fill(0); this.stress.fill(0); this.generation=0; }
  randomize(prob=0.22){ for(let i=0;i<this.size;i++){ this.alive[i]=Math.random()<prob?1:0; this.species[i]=(Math.random()*4)|0; this.energy[i]=0.4+Math.random()*0.6; } }
  pulseAnomaly(x,y,r=4,power=0.6){ for(let dy=-r;dy<=r;dy++) for(let dx=-r;dx<=r;dx++){ const d=Math.hypot(dx,dy); if(d<=r){ const i=this.idx(x+dx,y+dy); this.anomalyField[i]+=power*(1-d/r); } } }
  step(rule=DEFAULT_RULE,{surprise=0}={}){
    this.previousAlive.set(this.alive);
    if (surprise>0.25) for(let i=0;i<this.size;i++) this.anomalyField[i]+=surprise*0.03;
    for(let y=0;y<this.height;y++) for(let x=0;x<this.width;x++){
      const i=this.idx(x,y); const alive=this.alive[i]===1; const neighbors=this.countNeighbors(x,y);
      const energyBoost=Math.min(0.2,this.energy[i]*0.18); const stressPenalty=this.stress[i]*0.08;
      let born=rule.birth.includes(neighbors);
      if (!alive && born && Math.random()<Math.max(0,rule.mutation-energyBoost*0.08)) born=Math.random()>0.5;
      let survives=rule.survive.includes(neighbors);
      if (alive && survives) survives = Math.random() > stressPenalty;
      let next = alive ? (survives?1:0) : (born && this.energy[i]>0.18 ? 1:0);
      if (next && !alive){ this.energy[i]=Math.max(0,this.energy[i]-0.12); this.species[i]=(neighbors%4)+1; }
      if (next){ this.age[i]=Math.min(65535,this.age[i]+1); this.memory[i]=Math.min(1,this.memory[i]+0.05); this.memoryField[i]=Math.min(1,this.memoryField[i]+0.07); this.signalField[i]=Math.min(1,this.signalField[i]+0.1); }
      else { this.age[i]=0; this.memory[i]*=0.98; this.signalField[i]*=0.9; }
      this.stress[i]=Math.max(0, this.stress[i]*0.95 + this.anomalyField[i]*0.06 + Math.abs(neighbors-3)*0.004);
      this.nextAlive[i]=next;
    }
    this.alive.set(this.nextAlive);
    this.fieldDynamics();
    this.generation++;
  }
  fieldDynamics(){
    for(let i=0;i<this.size;i++){
      this.energy[i]+= this.alive[i] ? -0.008 : 0.004;
      this.energy[i]=Math.min(1,Math.max(0.05,this.energy[i]));
      this.anomalyField[i]*=0.93; this.memoryField[i]*=0.965;
    }
  }
  applyReseed(){
    for(let i=0;i<this.size;i++){
      if (Math.random()<0.03){ this.alive[i]=1; this.energy[i]=0.7; this.species[i]=((Math.random()*4)|0)+1; }
    }
  }
  snapshot(){ return { width:this.width, height:this.height, generation:this.generation, alive:this.alive.slice(), age:this.age, energy:this.energy, stress:this.stress, memory:this.memory }; }
}
