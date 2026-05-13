const TRAIT_NAMES = [
  'birthBias',
  'surviveBias',
  'energyUptake',
  'stressTolerance',
  'signalEmission',
  'memoryAffinity',
  'mutationRate',
  'cohesion',
  'mobilityBias',
  'colorHue',
];

const DEFAULT_TRAITS = {
  birthBias: 0.5,
  surviveBias: 0.5,
  energyUptake: 0.52,
  stressTolerance: 0.48,
  signalEmission: 0.42,
  memoryAffinity: 0.45,
  mutationRate: 0.035,
  cohesion: 0.5,
  mobilityBias: 0.36,
  colorHue: 0.52,
};

const clamp01 = value => Math.max(0, Math.min(1, value));

function normalizeTraits(input = {}) {
  const traits = {};
  for (const name of TRAIT_NAMES) traits[name] = clamp01(input[name] ?? DEFAULT_TRAITS[name]);
  traits.mutationRate = Math.min(0.24, traits.mutationRate);
  return traits;
}

function averageTraits(genomes) {
  const traits = {};
  for (const name of TRAIT_NAMES) traits[name] = 0;
  for (const genome of genomes) {
    for (const name of TRAIT_NAMES) traits[name] += genome.traits[name];
  }
  for (const name of TRAIT_NAMES) traits[name] /= Math.max(1, genomes.length);
  return normalizeTraits(traits);
}

function traitsKey(traits) {
  return TRAIT_NAMES.map(name => Math.round((traits[name] ?? 0) * 20)).join(':');
}

export class GenomeRegistry {
  constructor() {
    this.reset();
  }

  reset() {
    this.genomes = new Map();
    this.lineages = new Map();
    this.reuse = new Map();
    this.maxGenomes = 4096;
    this.nextGenomeId = 1;
    this.nextLineageId = 1;
  }

  createFounderGenome(seedTraits = {}, metadata = {}) {
    const lineageId = metadata.lineageId || this.nextLineageId++;
    const id = this.nextGenomeId++;
    const genome = {
      id,
      lineageId,
      parentGenomeIds: [],
      generationBorn: metadata.generation ?? 0,
      traits: normalizeTraits(seedTraits),
      births: 0,
      mutations: 0,
    };
    this.genomes.set(id, genome);
    this.reuse.set(traitsKey(genome.traits), id);
    this.lineages.set(lineageId, {
      id: lineageId,
      founderGenomeId: id,
      parentLineageIds: metadata.parentLineageIds || [],
      generationBorn: metadata.generation ?? 0,
      births: 0,
      extinctions: 0,
    });
    return id;
  }

  get(id) {
    return this.genomes.get(id) || this.genomes.get(1) || null;
  }

  ensureFounderForSpecies(species = 1, generation = 0) {
    const hue = ((species * 0.137) + 0.38) % 1;
    return this.createFounderGenome({
      birthBias: 0.42 + (species % 3) * 0.08,
      surviveBias: 0.46 + (species % 4) * 0.05,
      energyUptake: 0.48 + (species % 5) * 0.06,
      stressTolerance: 0.42 + (species % 4) * 0.07,
      signalEmission: 0.34 + (species % 5) * 0.08,
      memoryAffinity: 0.36 + (species % 4) * 0.08,
      mutationRate: 0.025 + (species % 3) * 0.012,
      cohesion: 0.42 + (species % 4) * 0.08,
      mobilityBias: 0.28 + (species % 5) * 0.08,
      colorHue: hue,
    }, { generation });
  }

  inherit(parentGenomeIds, localContext = {}) {
    const parents = parentGenomeIds.map(id => this.get(id)).filter(Boolean);
    if (!parents.length) return { genomeId: this.createFounderGenome({}, localContext), lineageId: this.nextLineageId - 1, mutated: true };

    const baseTraits = averageTraits(parents);
    const baseMutation = parents.reduce((sum, genome) => sum + genome.traits.mutationRate, 0) / parents.length;
    const mutationPressure = clamp01(localContext.mutationPressure ?? 0);
    const shouldMutate = Math.random() < Math.min(0.72, baseMutation + mutationPressure);
    const traits = shouldMutate ? this.mutateTraits(baseTraits, mutationPressure) : normalizeTraits(baseTraits);
    const parentGenomeIdsUnique = [...new Set(parents.map(genome => genome.id))];
    const dominantParent = parents[0];

    let lineageId = dominantParent.lineageId;
    const newLineagePressure = mutationPressure + (localContext.novelty ?? 0) * 0.35;
    if (shouldMutate && Math.random() < Math.min(0.28, newLineagePressure)) {
      lineageId = this.nextLineageId++;
      this.lineages.set(lineageId, {
        id: lineageId,
        founderGenomeId: null,
        parentLineageIds: [...new Set(parents.map(genome => genome.lineageId))],
        generationBorn: localContext.generation ?? 0,
        births: 0,
        extinctions: 0,
      });
    }

    const key = traitsKey(traits);
    let genomeId = this.reuse.get(key) || null;
    if (!genomeId && this.genomes.size >= this.maxGenomes) {
      genomeId = this.findNearestGenome(traits, lineageId)?.id || null;
    }
    if (!genomeId) {
      genomeId = this.nextGenomeId++;
      this.genomes.set(genomeId, {
        id: genomeId,
        lineageId,
        parentGenomeIds: parentGenomeIdsUnique,
        generationBorn: localContext.generation ?? 0,
        traits,
        births: 0,
        mutations: shouldMutate ? 1 : 0,
      });
      this.reuse.set(key, genomeId);
      const lineage = this.lineages.get(lineageId);
      if (lineage && !lineage.founderGenomeId) lineage.founderGenomeId = genomeId;
    }

    for (const genome of parents) genome.births++;
    const lineage = this.lineages.get(lineageId);
    if (lineage) lineage.births++;
    return { genomeId, lineageId, mutated: shouldMutate };
  }

  findNearestGenome(traits, preferredLineageId = 0) {
    let best = null;
    let bestDistance = Infinity;
    for (const genome of this.genomes.values()) {
      let distance = genome.lineageId === preferredLineageId ? -0.08 : 0;
      for (const name of TRAIT_NAMES) distance += Math.abs((traits[name] ?? 0) - genome.traits[name]);
      if (distance < bestDistance) {
        best = genome;
        bestDistance = distance;
      }
    }
    return best;
  }

  mutateTraits(traits, mutationPressure = 0) {
    const next = { ...traits };
    const amplitude = 0.035 + mutationPressure * 0.16;
    for (const name of TRAIT_NAMES) {
      const drift = (Math.random() * 2 - 1) * amplitude;
      next[name] = clamp01(next[name] + drift);
    }
    next.mutationRate = Math.min(0.24, next.mutationRate);
    return normalizeTraits(next);
  }

  exportGenomes() {
    return [...this.genomes.values()].map(genome => ({
      ...genome,
      traits: { ...genome.traits },
      parentGenomeIds: [...genome.parentGenomeIds],
    }));
  }

  exportLineages() {
    return [...this.lineages.values()].map(lineage => ({
      ...lineage,
      parentLineageIds: [...lineage.parentLineageIds],
    }));
  }

  importState({ genomes = [], lineages = [], nextGenomeId, nextLineageId } = {}) {
    this.reset();
    for (const lineage of lineages) this.lineages.set(lineage.id, { ...lineage, parentLineageIds: [...(lineage.parentLineageIds || [])] });
    for (const genome of genomes) {
      const normalized = {
        ...genome,
        parentGenomeIds: [...(genome.parentGenomeIds || [])],
        traits: normalizeTraits(genome.traits || {}),
      };
      this.genomes.set(normalized.id, normalized);
      this.reuse.set(traitsKey(normalized.traits), normalized.id);
    }
    this.maxGenomes = Math.max(4096, this.genomes.size);
    this.nextGenomeId = nextGenomeId || Math.max(1, ...this.genomes.keys()) + 1;
    this.nextLineageId = nextLineageId || Math.max(1, ...this.lineages.keys()) + 1;
  }
}

export { TRAIT_NAMES, DEFAULT_TRAITS, normalizeTraits };
