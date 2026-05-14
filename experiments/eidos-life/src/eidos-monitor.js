export const RULE_PRESETS = {
  GREEN:  { birth: [3], survive: [2,3], mutation: 0.0002 },
  AMBER:  { birth: [3,6], survive: [2,3], mutation: 0.0015 },
  RED:    { birth: [3], survive: [1,2,3], mutation: 0.0004, reseed: true },
  BLUE:   { birth: [3,4], survive: [2,3,4], mutation: 0.0008 },
  VIOLET: { birth: [3,5,6], survive: [2,3,4], mutation: 0.0030 },
  CALIBRATING: { birth:[3], survive:[2,3], mutation:0.0001 }
};

export class EidosMonitor {
  constructor(){ this.prevAlive=null; this.prevEntropy=0; this.regime='CALIBRATING'; this.timeline=[]; this.redStreak=0; }
  analyze({alive,age,energy,stress,novelty,generation}){
    const size=alive.length; let live=0, ageSum=0, en=0, st=0, flips=0;
    for(let i=0;i<size;i++){ live+=alive[i]; ageSum+=age[i]; en+=energy[i]; st+=stress[i]; if(this.prevAlive) flips += alive[i]===this.prevAlive[i]?0:1; }
    const aliveRatio=live/size;
    const entropy = aliveRatio===0||aliveRatio===1 ? 0 : -(aliveRatio*Math.log2(aliveRatio)+(1-aliveRatio)*Math.log2(1-aliveRatio));
    const surprise = this.prevAlive ? flips/size : 0;
    const compressionRatio = 1 + (1-aliveRatio) * 0.9 + (1-entropy) * 0.5;
    const plasticity = Math.max(0,Math.min(1,surprise + Math.abs(entropy-this.prevEntropy)*0.8 + novelty*0.4));
    const collapseRisk = aliveRatio<0.04 || aliveRatio>0.7 ? 1 : 0;
    let rawRegime='GREEN';
    if(generation<20) rawRegime='CALIBRATING';
    else if(collapseRisk) rawRegime='RED';
    else if(novelty>0.7 && surprise>0.12) rawRegime='VIOLET';
    else if(surprise>0.11) rawRegime='AMBER';
    else if(entropy>0.92) rawRegime='BLUE';

    this.redStreak = rawRegime === 'RED' ? this.redStreak + 1 : 0;
    const confirmedRegime = this.redStreak >= 3 ? 'RED' : (rawRegime === 'RED' ? 'AMBER' : rawRegime);
    const redFlicker = rawRegime === 'RED' && this.redStreak < 3 ? 1 : 0;

    this.regime=confirmedRegime; this.timeline.push(confirmedRegime); if(this.timeline.length>256) this.timeline.shift();
    const metrics={ generation, regime: confirmedRegime, rawRegime, confirmedRegime, redFlicker, surprise, entropy, compressionRatio, novelty, collapseRisk, plasticity, aliveRatio,
      meanAge: ageSum/size, meanEnergy: en/size, meanStress: st/size };
    this.prevAlive = alive.slice(); this.prevEntropy = entropy;
    return { metrics, rulePreset: RULE_PRESETS[rawRegime] };
  }
}
