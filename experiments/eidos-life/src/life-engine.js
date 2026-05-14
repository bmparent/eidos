import { GenomeRegistry } from './genome.js';
import { EcologyFields } from './ecology-fields.js';
import { LocalRegimeMap } from './local-regimes.js';

export const DEFAULT_RULE = { birth: [3], survive: [2, 3], mutation: 0.0002, reseed: false };
const clamp01 = value => Math.max(0, Math.min(1, value));
const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

export class LifeEngine {
  constructor({ width = 72, height = 72, evolutionEnabled = false, seed = 42, config = {} } = {}) {
    this.width = width; this.height = height; this.size = width * height; this.evolutionEnabled = evolutionEnabled; this.seedValue = seed >>> 0;
    this.rng = this.mulberry32(this.seedValue);
    this.config = { higgs_enabled: true, mass_base: 1, mass_phi_gain: 1, movement_mass_cost: 0.015, mutation_mass_cost: 0.025, reproduction_mass_cost: 0.05, higgs_decay: 0.001, higgs_diffusion: 0.02, higgs_min: 0, higgs_max: 3, max_events_in_state: 128, ...config };
    this.alive = new Uint8Array(this.size); this.previousAlive = new Uint8Array(this.size); this.nextAlive = new Uint8Array(this.size);
    this.age = new Uint16Array(this.size); this.energy = new Float32Array(this.size).fill(0.65); this.health = new Float32Array(this.size).fill(1);
    this.stress = new Float32Array(this.size); this.memory = new Float32Array(this.size); this.familiarity = new Float32Array(this.size);
    this.species = new Uint8Array(this.size); this.genomeId = new Uint32Array(this.size); this.nextGenomeId = new Uint32Array(this.size);
    this.lineageId = new Uint32Array(this.size); this.nextLineageId = new Uint32Array(this.size); this.reproductionCount = new Uint16Array(this.size);
    this.nutrientField = new Float32Array(this.size).fill(0.72); this.toxicityField = new Float32Array(this.size); this.signalField = new Float32Array(this.size);
    this.heatField = new Float32Array(this.size); this.memoryField = new Float32Array(this.size); this.higgsPhiField = new Float32Array(this.size);
    this.anomalyField = this.heatField; this.wasteField = this.toxicityField;
    this.lastAction = new Uint8Array(this.size); this.generation = 0; this.lastBirthCount = 0; this.lastDeathCount = 0; this.lastMutationCount = 0;
    this.genomeRegistry = new GenomeRegistry(); this.ecology = new EcologyFields(width, height); this.localRegimes = new LocalRegimeMap(width, height);
    this.events = []; this.regimeCounts = { GREEN: 0, AMBER: 0, RED: 0, BLUE: 0, VIOLET: 0 };
    this.clear();
  }
  mulberry32(a){return ()=>{let t=a+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
  idx(x,y){return((y+this.height)%this.height)*this.width+((x+this.width)%this.width);} countNeighbors(x,y){let n=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;n+=this.alive[this.idx(x+dx,y+dy)];}return n;}
  clear(){this.alive.fill(0);this.previousAlive.fill(0);this.nextAlive.fill(0);this.age.fill(0);this.energy.fill(0.65);this.health.fill(1);this.stress.fill(0);this.memory.fill(0);this.familiarity.fill(0);this.species.fill(0);this.genomeId.fill(0);this.lineageId.fill(0);this.nextGenomeId.fill(0);this.nextLineageId.fill(0);this.reproductionCount.fill(0);this.generation=0;this.events=[];this.genomeRegistry.reset();this.ecology.reset(this);}
  randomize(prob=0.22){for(let i=0;i<this.size;i++){const alive=this.rng()<prob?1:0;this.alive[i]=alive;this.health[i]=0.7+this.rng()*0.3;if(alive){this.energy[i]=0.4+this.rng()*0.6;this.species[i]=((this.rng()*5)|0)+1;this.assignGenome(i,this.species[i]);} this.higgsPhiField[i]=this.config.higgs_enabled?0.3+this.rng()*1.3:0;}}
  assignGenome(i,species=1){const founder=this.genomeRegistry.ensureFounderForSpecies(species,this.generation);this.species[i]=species;this.genomeId[i]=founder;this.lineageId[i]=this.genomeRegistry.get(founder)?.lineageId||species;}

  seed(points){this.clear();for(const [x,y,s=1] of points)this.setAliveCell(this.idx(x,y),s);}
  setAliveCell(i,s=1){this.alive[i]=1;this.energy[i]=Math.max(this.energy[i],0.45);this.health[i]=Math.max(this.health[i],0.7);this.assignGenome(i,s);}
  computeMass(i){if(!this.config.higgs_enabled) return this.config.mass_base; return this.config.mass_base + (this.config.mass_phi_gain * this.higgsPhiField[i]);}
  step(rule=DEFAULT_RULE,options={}){
    const evolutionEnabled=options.evolutionEnabled??this.evolutionEnabled;this.previousAlive.set(this.alive);this.lastBirthCount=0;this.lastDeathCount=0;this.lastMutationCount=0;
    for(let y=0;y<this.height;y++)for(let x=0;x<this.width;x++){const i=this.idx(x,y);const isAlive=this.alive[i]===1;const neighbors=this.countNeighbors(x,y);isAlive?this.stepAlive(i,neighbors,rule,evolutionEnabled):this.stepDead(i,neighbors,rule,evolutionEnabled);}
    this.alive.set(this.nextAlive);this.genomeId.set(this.nextGenomeId);this.lineageId.set(this.nextLineageId);this.ecology.update(this,this.genomeRegistry,{intervention:options.intervention||'passive'});this.generation++;
  }
  stepAlive(i,neighbors,rule,evolutionEnabled){const mass=this.computeMass(i);const metabolism=(0.008+this.stress[i]*0.01)*(1+mass*0.15);this.energy[i]=clamp01(this.energy[i]-metabolism);
    const feed=Math.min(this.nutrientField[i],0.04);this.energy[i]=clamp01(this.energy[i]+feed);this.nutrientField[i]=Math.max(0,this.nutrientField[i]-feed);
    this.health[i]=clamp01(this.health[i]-this.toxicityField[i]*0.035); if(this.health[i]<0.65&&this.energy[i]>0.3){const repair=0.02;this.energy[i]-=repair;this.health[i]=clamp01(this.health[i]+repair*0.9);this.lastAction[i]=4;}
    const survives=rule.survive.includes(neighbors)&&this.energy[i]>0.02&&this.health[i]>0.02; if(!survives){this.die(i,['energy/health collapse']);return;}
    this.nextAlive[i]=1; this.nextGenomeId[i]=this.genomeId[i]; this.nextLineageId[i]=this.lineageId[i]; this.age[i]=Math.min(65535,this.age[i]+1);
    const canReproduce=evolutionEnabled&&this.energy[i]>0.8&&this.health[i]>0.5&&neighbors<5; if(canReproduce && this.rng()<(0.22/(1+mass*this.config.reproduction_mass_cost))){this.tryReproduce(i,mass);}
    this.familiarity[i]=clamp01(this.familiarity[i]*0.98+this.memoryField[i]*0.04); this.memory[i]=clamp01(this.memory[i]*0.97+0.03);
  }
  collectParentGenomeIdsByIndex(i){const x=i%this.width;const y=Math.floor(i/this.width);const ids=[];for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;const j=this.idx(x+dx,y+dy);if(this.alive[j]&&this.genomeId[j])ids.push(this.genomeId[j]);}return ids.slice(0,4);}
  stepDead(i,neighbors,rule){const born=rule.birth.includes(neighbors)&&this.energy[i]>0.1&&this.rng()>this.toxicityField[i]*0.5; if(!born){this.nextAlive[i]=0;this.nextGenomeId[i]=0;this.nextLineageId[i]=0;return;}
    const parentIds=this.collectParentGenomeIdsByIndex(i);
    this.nextAlive[i]=1;this.energy[i]=Math.max(0.32,this.energy[i]);this.health[i]=0.65;
    if(parentIds.length){const child=this.genomeRegistry.inherit(parentIds,{generation:this.generation,mutationPressure:0.15});this.genomeId[i]=child.genomeId;this.lineageId[i]=child.lineageId;this.species[i]=Math.max(1,child.genomeId%255);if(child.mutated)this.lastMutationCount++;}
    else { this.assignGenome(i,(neighbors%4)+1); }
    this.nextGenomeId[i]=this.genomeId[i];this.nextLineageId[i]=this.lineageId[i];this.lastBirthCount++;
  }
  tryReproduce(i,mass){for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;const j=this.idx((i%this.width)+dx,Math.floor(i/this.width)+dy);if(this.nextAlive[j])continue;
      this.nextAlive[j]=1;this.nextGenomeId[j]=this.genomeId[i];this.nextLineageId[j]=this.lineageId[i];this.energy[j]=this.energy[i]*0.35;this.health[j]=0.7;this.age[j]=0;this.energy[i]-=0.22*(1+mass*this.config.reproduction_mass_cost);
      const mutChance=0.03/(1+mass*this.config.mutation_mass_cost);if(this.rng()<mutChance){this.lastMutationCount++;this.events.push({generation:this.generation,event_type:'mutation',severity:'AMBER'});} this.lastBirthCount++;this.reproductionCount[i]++;return;}
  }
  die(i,why=[]){this.nextAlive[i]=0;this.nextGenomeId[i]=0;this.nextLineageId[i]=0;this.lastDeathCount++;this.nutrientField[i]=clamp01(this.nutrientField[i]+0.2);this.toxicityField[i]=clamp01(this.toxicityField[i]+0.08);this.memoryField[i]=clamp01(this.memoryField[i]+0.12);this.events.push({generation:this.generation,event_type:'death',severity:'RED',organism_id:i,why});}
  snapshot(){return {width:this.width,height:this.height,generation:this.generation,alive:this.alive.slice(),age:this.age,energy:this.energy,stress:this.stress,memory:this.memory,species:this.species,genomeId:this.genomeId,lineageId:this.lineageId,signalField:this.signalField,anomalyField:this.anomalyField,memoryField:this.memoryField,nutrientField:this.nutrientField,wasteField:this.wasteField,toxicityField:this.toxicityField,heatField:this.heatField,higgsPhiField:this.higgsPhiField,health:this.health};}
  exportState({scenario='',settings={},detail='summary'}={}){const aliveCount=this.alive.reduce((a,b)=>a+b,0);const density=aliveCount/this.size;const mEnergy=[];const mHealth=[];const mMass=[];for(let i=0;i<this.size;i++)if(this.alive[i]){mEnergy.push(this.energy[i]);mHealth.push(this.health[i]);mMass.push(this.computeMass(i));}
    return {version:'0.4',life_version:'0.2-life-fields',scenario,settings,generation:this.generation,width:this.width,height:this.height,alive_count:aliveCount,density,regime:density<0.04?'RED':mean(mHealth)<0.45?'AMBER':'GREEN',grid:Array.from(this.alive),alive:Array.from(this.alive),fields_available:['nutrient','toxicity','signal','heat','memory_residue','higgs_phi'],births:this.lastBirthCount,deaths:this.lastDeathCount,mutations:this.lastMutationCount,global_energy_mean:mean(mEnergy),global_health_mean:mean(mHealth),global_mass_mean:mean(mMass),global_phi_mean:mean(Array.from(this.higgsPhiField)),global_phi_std:0,global_toxicity_mean:mean(Array.from(this.toxicityField)),global_nutrient_mean:mean(Array.from(this.nutrientField)),memory_residue_mean:mean(Array.from(this.memoryField)),novelty_mean:mean(Array.from(this.stress)),familiarity_mean:mean(Array.from(this.familiarity)),event_counts:{recent:this.events.length},latest_events:this.events.slice(-16),genomeRegistry:{genomes:this.genomeRegistry.exportGenomes(),lineages:this.genomeRegistry.exportLineages(),nextGenomeId:this.genomeRegistry.nextGenomeId,nextLineageId:this.genomeRegistry.nextLineageId},field_stats:{nutrient_mean:mean(Array.from(this.nutrientField)),toxicity_mean:mean(Array.from(this.toxicityField))},...(detail==='fields'?{fields:{nutrient:Array.from(this.nutrientField),toxicity:Array.from(this.toxicityField),higgs_phi:Array.from(this.higgsPhiField),memory_residue:Array.from(this.memoryField),heat:Array.from(this.heatField)}}:{}),...(detail==='organisms'?{organisms_sample:this.sampleOrganisms(32)}:{})};}

  pulseAnomaly(x,y,r=4,power=0.6){for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const d=Math.hypot(dx,dy);if(d<=r){const i=this.idx(x+dx,y+dy);this.heatField[i]=clamp01(this.heatField[i]+power*(1-d/r));}}}
  applyEidosIntervention(){ }
  importState(state){if(!state||state.width!==this.width||state.height!==this.height) throw new Error('World state dimensions do not match this engine.');for (const [k,a] of [['alive',this.alive],['age',this.age],['energy',this.energy],['stress',this.stress],['memory',this.memory],['genomeId',this.genomeId],['lineageId',this.lineageId],['nutrientField',this.nutrientField],['toxicityField',this.toxicityField],['signalField',this.signalField],['heatField',this.heatField],['memoryField',this.memoryField],['higgsPhiField',this.higgsPhiField],['health',this.health]]) if(state[k]) a.set(state[k].slice(0,a.length));this.generation=state.generation||0;this.genomeRegistry.importState(state.genomeRegistry||{});}

  sampleOrganisms(limit=16){const rows=[];for(let i=0;i<this.size&&rows.length<limit;i++)if(this.alive[i])rows.push({id:`cell_${i}`,lineage_id:`lineage_${this.lineageId[i]}`,alive:true,x:i%this.width,y:Math.floor(i/this.width),age:this.age[i],energy:this.energy[i],health:this.health[i],mass:this.computeMass(i),field_coupling:1,mutation_rate:0.03,plasticity:1/(1+this.computeMass(i)),memory_strength:this.memory[i],regime:this.energy[i]<0.2?'RED':'GREEN',last_action:'rest',reproduction_count:this.reproductionCount[i],genome:{mutation_rate:0.03,field_coupling:1}});return rows;}
}
