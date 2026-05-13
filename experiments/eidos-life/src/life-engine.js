import { GenomeRegistry } from './genome.js';
import { EcologyFields } from './ecology-fields.js';
import { LocalRegimeMap } from './local-regimes.js';

export const DEFAULT_RULE = { birth: [3], survive: [2, 3], mutation: 0.0002, reseed: false };

const clamp01 = value => Math.max(0, Math.min(1, value));

export class LifeEngine {
  constructor({ width = 72, height = 72, evolutionEnabled = false } = {}) {
    this.width = width;
    this.height = height;
    this.size = width * height;
    this.evolutionEnabled = evolutionEnabled;
    this.alive = new Uint8Array(this.size);
    this.previousAlive = new Uint8Array(this.size);
    this.nextAlive = new Uint8Array(this.size);
    this.age = new Uint16Array(this.size);
    this.energy = new Float32Array(this.size).fill(0.65);
    this.species = new Uint8Array(this.size);
    this.genomeId = new Uint32Array(this.size);
    this.nextGenomeId = new Uint32Array(this.size);
    this.lineageId = new Uint32Array(this.size);
    this.nextLineageId = new Uint32Array(this.size);
    this.memory = new Float32Array(this.size);
    this.stress = new Float32Array(this.size);
    this.signalField = new Float32Array(this.size);
    this.anomalyField = new Float32Array(this.size);
    this.memoryField = new Float32Array(this.size);
    this.nutrientField = new Float32Array(this.size).fill(0.72);
    this.wasteField = new Float32Array(this.size);
    this.generation = 0;
    this.genomeRegistry = new GenomeRegistry();
    this.ecology = new EcologyFields(width, height);
    this.localRegimes = new LocalRegimeMap(width, height);
    this.founderBySpecies = new Map();
    this.lastBirthCount = 0;
    this.lastDeathCount = 0;
    this.lastMutationCount = 0;
    this.lastLocalRegimeUpdate = { violetCount: 0, violetEmergence: false };
  }

  idx(x, y) {
    return ((y + this.height) % this.height) * this.width + ((x + this.width) % this.width);
  }

  countNeighbors(x, y) {
    let n = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        n += this.alive[this.idx(x + dx, y + dy)];
      }
    }
    return n;
  }

  seed(points) {
    this.clear();
    for (const [x, y, s = 1] of points) this.setAliveCell(this.idx(x, y), s);
  }

  clear() {
    this.alive.fill(0);
    this.previousAlive.fill(0);
    this.nextAlive.fill(0);
    this.age.fill(0);
    this.energy.fill(0.65);
    this.species.fill(0);
    this.genomeId.fill(0);
    this.nextGenomeId.fill(0);
    this.lineageId.fill(0);
    this.nextLineageId.fill(0);
    this.memory.fill(0);
    this.stress.fill(0);
    this.generation = 0;
    this.genomeRegistry.reset();
    this.founderBySpecies.clear();
    this.ecology.reset(this);
    this.localRegimes.reset();
    this.lastBirthCount = 0;
    this.lastDeathCount = 0;
    this.lastMutationCount = 0;
  }

  randomize(prob = 0.22) {
    for (let i = 0; i < this.size; i++) {
      const alive = Math.random() < prob ? 1 : 0;
      this.alive[i] = alive;
      this.energy[i] = 0.4 + Math.random() * 0.6;
      if (alive) {
        const species = ((Math.random() * 5) | 0) + 1;
        this.assignGenome(i, species);
      } else {
        this.species[i] = 0;
        this.genomeId[i] = 0;
        this.lineageId[i] = 0;
      }
    }
  }

  setAliveCell(i, species = 1) {
    this.alive[i] = 1;
    this.energy[i] = Math.max(this.energy[i], 0.45);
    this.assignGenome(i, species);
  }

  assignGenome(i, species = 1) {
    if (!this.founderBySpecies.has(species)) {
      this.founderBySpecies.set(species, this.genomeRegistry.ensureFounderForSpecies(species, this.generation));
    }
    const genomeId = this.founderBySpecies.get(species);
    const genome = this.genomeRegistry.get(genomeId);
    this.species[i] = species;
    this.genomeId[i] = genomeId;
    this.lineageId[i] = genome?.lineageId || species;
  }

  pulseAnomaly(x, y, r = 4, power = 0.6) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const d = Math.hypot(dx, dy);
        if (d <= r) {
          const i = this.idx(x + dx, y + dy);
          this.anomalyField[i] = clamp01(this.anomalyField[i] + power * (1 - d / r));
        }
      }
    }
  }

  step(rule = DEFAULT_RULE, options = {}) {
    const evolutionEnabled = options.evolutionEnabled ?? this.evolutionEnabled;
    const surprise = options.surprise || 0;
    const mutationPressureSetting = options.mutationPressure || 'adaptive';
    const intervention = options.intervention || 'passive';
    this.previousAlive.set(this.alive);
    this.lastBirthCount = 0;
    this.lastDeathCount = 0;
    this.lastMutationCount = 0;

    if (evolutionEnabled) {
      this.lastLocalRegimeUpdate = this.localRegimes.update(this.snapshot(), { novelty: options.novelty || 0, generation: this.generation });
    }
    if (surprise > 0.25) {
      for (let i = 0; i < this.size; i++) this.anomalyField[i] = clamp01(this.anomalyField[i] + surprise * 0.03);
    }

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const i = this.idx(x, y);
        const isAlive = this.alive[i] === 1;
        const neighbors = this.countNeighbors(x, y);
        if (evolutionEnabled) this.stepEvolutionCell(i, x, y, isAlive, neighbors, rule, mutationPressureSetting);
        else this.stepBaselineCell(i, isAlive, neighbors, rule);
      }
    }

    this.alive.set(this.nextAlive);
    this.genomeId.set(this.nextGenomeId);
    this.lineageId.set(this.nextLineageId);
    if (evolutionEnabled) this.ecology.update(this, this.genomeRegistry, { intervention, collapseRisk: options.collapseRisk || 0 });
    else this.fieldDynamics();
    this.generation++;
  }

  stepBaselineCell(i, isAlive, neighbors, rule) {
    const energyBoost = Math.min(0.2, this.energy[i] * 0.18);
    const stressPenalty = this.stress[i] * 0.08;
    let born = rule.birth.includes(neighbors);
    if (!isAlive && born && Math.random() < Math.max(0, rule.mutation - energyBoost * 0.08)) born = Math.random() > 0.5;
    let survives = rule.survive.includes(neighbors);
    if (isAlive && survives) survives = Math.random() > stressPenalty;
    const next = isAlive ? (survives ? 1 : 0) : (born && this.energy[i] > 0.18 ? 1 : 0);
    this.writeNextCell(i, next, isAlive, null);
    if (next && !isAlive) this.species[i] = (neighbors % 4) + 1;
  }

  stepEvolutionCell(i, x, y, isAlive, neighbors, rule, mutationPressureSetting) {
    const currentGenome = this.genomeRegistry.get(this.genomeId[i]);
    const modifier = this.localRegimes.modifierAtIndex(i, this.width);
    const context = this.ecology.contextAt(this, i, currentGenome, modifier);
    const trait = currentGenome?.traits || {};
    const stressPenalty = this.stress[i] * (0.15 - (trait.stressTolerance || 0.5) * 0.09);
    const memorySupport = this.memoryField[i] * (trait.memoryAffinity || 0.5) * 0.08;
    const cohesionSupport = this.relatedNeighborRatio(x, y, this.lineageId[i]) * (trait.cohesion || 0.5) * 0.06;
    let survives = rule.survive.includes(neighbors);
    if (isAlive) {
      const surviveScore = (survives ? 0.84 : 0.28) + (trait.surviveBias - 0.5) * 0.28 + context.nutrient * 0.08 + memorySupport + cohesionSupport + (modifier.surviveBoost || 0) - stressPenalty - context.waste * 0.08;
      survives = Math.random() < clamp01(surviveScore);
      this.writeNextCell(i, survives ? 1 : 0, true, currentGenome);
      if (!survives) this.lastDeathCount++;
      return;
    }

    let born = rule.birth.includes(neighbors);
    const parentIds = this.collectParentGenomeIds(x, y);
    const parentTraits = this.averageParentTraits(parentIds);
    const birthScore = (born ? 0.76 : 0.08) + (parentTraits.birthBias - 0.5) * 0.32 + context.nutrient * 0.18 + this.signalField[i] * 0.08 + memorySupport + (modifier.birthBoost || 0) - context.waste * 0.12 - context.anomaly * 0.05;
    born = parentIds.length > 0 && this.energy[i] > 0.12 && Math.random() < clamp01(birthScore);
    if (!born) {
      this.writeNextCell(i, 0, false, null);
      return;
    }

    const settingPressure = mutationPressureSetting === 'high' ? 0.12 : mutationPressureSetting === 'medium' ? 0.055 : mutationPressureSetting === 'low' ? -0.015 : 0;
    const child = this.genomeRegistry.inherit(parentIds, {
      generation: this.generation,
      mutationPressure: clamp01(context.mutationPressure + settingPressure + (modifier.mutationBoost || 0)),
      novelty: context.stability < 0.2 ? 0.3 : 0,
    });
    this.nextAlive[i] = 1;
    this.nextGenomeId[i] = child.genomeId;
    this.nextLineageId[i] = child.lineageId;
    this.age[i] = 1;
    this.memory[i] = Math.min(1, this.memory[i] + 0.04);
    this.memoryField[i] = Math.min(1, this.memoryField[i] + 0.08);
    this.signalField[i] = Math.min(1, this.signalField[i] + 0.08);
    this.energy[i] = Math.max(0.05, this.energy[i] - 0.09 - context.nutrient * 0.03);
    this.species[i] = Math.max(1, child.genomeId % 255);
    this.lastBirthCount++;
    if (child.mutated) this.lastMutationCount++;
  }

  writeNextCell(i, next, wasAlive, genome) {
    this.nextAlive[i] = next;
    if (next) {
      if (wasAlive) {
        this.nextGenomeId[i] = this.genomeId[i];
        this.nextLineageId[i] = this.lineageId[i];
      } else if (genome) {
        this.nextGenomeId[i] = genome.id;
        this.nextLineageId[i] = genome.lineageId;
      }
      this.age[i] = Math.min(65535, this.age[i] + 1);
      this.memory[i] = Math.min(1, this.memory[i] + 0.05);
      this.memoryField[i] = Math.min(1, this.memoryField[i] + 0.07);
      this.signalField[i] = Math.min(1, this.signalField[i] + 0.1);
    } else {
      this.nextGenomeId[i] = 0;
      this.nextLineageId[i] = 0;
      this.age[i] = 0;
      this.memory[i] *= 0.98;
      this.signalField[i] *= 0.9;
    }
    this.stress[i] = Math.max(0, this.stress[i] * 0.95 + this.anomalyField[i] * 0.06 + (this.wasteField[i] || 0) * 0.035);
  }

  collectParentGenomeIds(x, y) {
    const ids = [];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const i = this.idx(x + dx, y + dy);
        if (this.alive[i] && this.genomeId[i]) ids.push(this.genomeId[i]);
      }
    }
    ids.sort((a, b) => (this.genomeRegistry.get(b)?.traits.energyUptake || 0) - (this.genomeRegistry.get(a)?.traits.energyUptake || 0));
    return ids.slice(0, 4);
  }

  averageParentTraits(parentIds) {
    if (!parentIds.length) return { birthBias: 0.5 };
    let birthBias = 0;
    for (const id of parentIds) birthBias += this.genomeRegistry.get(id)?.traits.birthBias || 0.5;
    return { birthBias: birthBias / parentIds.length };
  }

  relatedNeighborRatio(x, y, lineageId) {
    if (!lineageId) return 0;
    let related = 0, live = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const i = this.idx(x + dx, y + dy);
        if (this.alive[i]) {
          live++;
          if (this.lineageId[i] === lineageId) related++;
        }
      }
    }
    return live ? related / live : 0;
  }

  fieldDynamics() {
    for (let i = 0; i < this.size; i++) {
      this.energy[i] += this.alive[i] ? -0.008 : 0.004;
      this.energy[i] = Math.min(1, Math.max(0.05, this.energy[i]));
      this.anomalyField[i] *= 0.93;
      this.memoryField[i] *= 0.965;
    }
  }

  applyReseed() {
    for (let i = 0; i < this.size; i++) {
      if (Math.random() < 0.03) {
        this.alive[i] = 1;
        this.energy[i] = 0.7;
        this.assignGenome(i, ((Math.random() * 5) | 0) + 1);
      }
    }
  }

  applyEidosIntervention(mode = 'passive', metrics = {}) {
    if (mode === 'passive') return;
    const rareLineages = new Set();
    if (mode === 'guardian' && metrics.collapseRisk) {
      for (let i = 0; i < this.size; i++) {
        this.nutrientField[i] = Math.min(1, this.nutrientField[i] + 0.01);
        this.stress[i] *= 0.96;
      }
      return;
    }
    if (mode === 'experimental') {
      for (let i = 0; i < this.size; i += 17) this.anomalyField[i] = Math.min(1, this.anomalyField[i] + 0.015);
    }
    for (let i = 0; i < this.size; i++) if (this.alive[i] && this.lineageId[i]) rareLineages.add(this.lineageId[i]);
    if (rareLineages.size > 6 && mode === 'guardian') {
      for (let i = 0; i < this.size; i++) {
        if (this.alive[i]) this.memoryField[i] = Math.min(1, this.memoryField[i] + 0.01);
      }
    }
  }

  snapshot() {
    return {
      width: this.width,
      height: this.height,
      generation: this.generation,
      alive: this.alive.slice(),
      age: this.age,
      energy: this.energy,
      stress: this.stress,
      memory: this.memory,
      species: this.species,
      genomeId: this.genomeId,
      lineageId: this.lineageId,
      signalField: this.signalField,
      anomalyField: this.anomalyField,
      memoryField: this.memoryField,
      nutrientField: this.nutrientField,
      wasteField: this.wasteField,
    };
  }

  exportState({ scenario = '', settings = {} } = {}) {
    return {
      version: '0.4',
      scenario,
      settings,
      generation: this.generation,
      width: this.width,
      height: this.height,
      alive: Array.from(this.alive),
      age: Array.from(this.age),
      energy: Array.from(this.energy),
      stress: Array.from(this.stress),
      memory: Array.from(this.memory),
      genomeId: Array.from(this.genomeId),
      lineageId: Array.from(this.lineageId),
      nutrientField: Array.from(this.nutrientField),
      wasteField: Array.from(this.wasteField),
      signalField: Array.from(this.signalField),
      anomalyField: Array.from(this.anomalyField),
      memoryField: Array.from(this.memoryField),
      genomeRegistry: {
        genomes: this.genomeRegistry.exportGenomes(),
        lineages: this.genomeRegistry.exportLineages(),
        nextGenomeId: this.genomeRegistry.nextGenomeId,
        nextLineageId: this.genomeRegistry.nextLineageId,
      },
      monitor: { generation: this.generation },
    };
  }

  importState(state) {
    if (!state || state.width !== this.width || state.height !== this.height) throw new Error('World state dimensions do not match this engine.');
    this.clear();
    this.generation = state.generation || 0;
    for (const [key, array] of [
      ['alive', this.alive], ['age', this.age], ['energy', this.energy], ['stress', this.stress], ['memory', this.memory],
      ['genomeId', this.genomeId], ['lineageId', this.lineageId], ['nutrientField', this.nutrientField], ['wasteField', this.wasteField],
      ['signalField', this.signalField], ['anomalyField', this.anomalyField], ['memoryField', this.memoryField],
    ]) {
      if (state[key]) array.set(state[key].slice(0, array.length));
    }
    this.genomeRegistry.importState(state.genomeRegistry || {});
  }
}
