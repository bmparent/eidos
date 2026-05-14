const clamp01 = value => Math.max(0, Math.min(1, value));

export class OrganismTracker {
  constructor({ maxHistory = 48, splitConfirmFrames = 3, mergeConfirmFrames = 3, deathConfirmFrames = 3, eventCooldownFrames = 10 } = {}) {
    this.maxHistory = maxHistory;
    this.nextId = 1;
    this.active = new Map();
    this.dead = [];
    this.lastEvents = [];
    this.lineageGraph = new Map();
    this.splitConfirmFrames = splitConfirmFrames; this.mergeConfirmFrames = mergeConfirmFrames; this.deathConfirmFrames = deathConfirmFrames; this.eventCooldownFrames = eventCooldownFrames;
    this.candidates = new Map(); this.cooldowns = new Map(); this.candidateEventTypeCounts = {}; this.confirmedEventTypeCounts = {}; this.recentCandidateEvents=[];
  }

  reset() {
    this.nextId = 1;
    this.active.clear();
    this.dead = [];
    this.lastEvents = [];
    this.lineageGraph.clear(); this.candidates.clear(); this.cooldowns.clear();
  }

  update(snapshot, metrics = {}, genomeRegistry = null) {
    const components = this.extractComponents(snapshot);
    const previous = [...this.active.values()];
    const usedPrevious = new Set();
    const prevMatches = new Map();
    const nextActive = new Map();
    const events = [];

    for (const component of components) {
      const candidates = previous
        .map(prev => ({ organism: prev, score: this.matchScore(prev, component) }))
        .filter(item => item.score < 9)
        .sort((a, b) => a.score - b.score);
      const best = candidates.find(item => !usedPrevious.has(item.organism.id)) || candidates[0] || null;
      const isNew = !best || usedPrevious.has(best.organism.id);
      const id = isNew ? this.nextId++ : best.organism.id;
      const parents = candidates.slice(0, 3).map(item => item.organism.id);
      const previousOrganism = isNew ? null : best.organism;

      if (previousOrganism) usedPrevious.add(previousOrganism.id);
      for (const parentId of parents) {
        if (!prevMatches.has(parentId)) prevMatches.set(parentId, []);
        prevMatches.get(parentId).push(id);
      }

      const organism = this.buildOrganism({
        id,
        component,
        snapshot,
        metrics,
        previous: previousOrganism,
        parentIds: isNew ? parents : previousOrganism.parentIds,
        genomeRegistry,
      });
      nextActive.set(id, organism);

      if (isNew) {
        events.push(event(snapshot.generation, 'organism_birth', 'low', organism, `Organism #${id} emerged in lineage ${organism.dominantLineageId}.`));
      }
      if (candidates.length > 1 && isNew) {
        events.push(event(snapshot.generation, 'organism_merge', 'medium', organism, `Organism #${id} formed from nearby organism merger pressure.`));
      }
      if (organism.ageFrames === 300 || organism.ageFrames === 900) {
        events.push(event(snapshot.generation, 'long_lived_organism', 'medium', organism, `Organism #${id} persisted for ${organism.ageFrames} frames.`));
      }
      if (previousOrganism && organism.mass > previousOrganism.mass + 24) {
        events.push(event(snapshot.generation, 'massive_growth', 'medium', organism, `Organism #${id} expanded from ${previousOrganism.mass} to ${organism.mass} cells.`));
      }
    }

    for (const prev of previous) {
      const matches = prevMatches.get(prev.id) || [];
      if (!matches.length) {
        const dead = { ...prev, status: 'dead', deathGeneration: snapshot.generation };
        this.dead.push(dead);
        events.push(event(snapshot.generation, 'organism_death', 'medium', dead, `Organism #${prev.id} died after ${prev.ageFrames} frames.`));
      } else if (matches.length > 1) {
        const parent = nextActive.get(matches[0]) || prev;
        events.push(event(snapshot.generation, 'organism_split', 'medium', parent, `Organism #${prev.id} split into ${matches.join(', ')}.`));
      }
    }

    this.active = nextActive;
    const confirmed=[];
    for (const e of events) {
      const key = `${e.type}:${e.organismId}`;
      const prev = this.candidates.get(key) || { frames: 0 };
      prev.frames += 1;
      this.candidates.set(key, prev);
      this.candidateEventTypeCounts[e.type] = (this.candidateEventTypeCounts[e.type] || 0) + 1;
      this.recentCandidateEvents.push(e); if (this.recentCandidateEvents.length>100) this.recentCandidateEvents.shift();
      const need = e.type==='organism_split'?this.splitConfirmFrames:e.type==='organism_merge'?this.mergeConfirmFrames:e.type==='organism_death'?this.deathConfirmFrames:1;
      const cooldownUntil = this.cooldowns.get(key) || -1;
      if (prev.frames >= need && snapshot.generation >= cooldownUntil) { confirmed.push(e); this.confirmedEventTypeCounts[e.type]=(this.confirmedEventTypeCounts[e.type]||0)+1; this.cooldowns.set(key, snapshot.generation + this.eventCooldownFrames); }
    }
    this.lastEvents = confirmed;
    this.updateLineageGraph();
    return this.getActiveOrganisms();
  }

  extractComponents({ alive, age, energy, stress, genomeId, lineageId, width, height }) {
    const visited = new Uint8Array(alive.length);
    const components = [];
    const idx = (x, y) => ((y + height) % height) * width + ((x + width) % width);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const start = idx(x, y);
        if (!alive[start] || visited[start]) continue;
        const stack = [[x, y]];
        const cells = [];
        let sx = 0, sy = 0, ageSum = 0, energySum = 0, stressSum = 0;
        let minX = width, maxX = 0, minY = height, maxY = 0;
        const genomeCounts = new Map();
        const lineageCounts = new Map();

        while (stack.length) {
          const [cx, cy] = stack.pop();
          const ci = idx(cx, cy);
          if (visited[ci] || !alive[ci]) continue;
          visited[ci] = 1;
          cells.push(ci);
          sx += cx;
          sy += cy;
          ageSum += age[ci] || 0;
          energySum += energy[ci] || 0;
          stressSum += stress[ci] || 0;
          minX = Math.min(minX, cx);
          maxX = Math.max(maxX, cx);
          minY = Math.min(minY, cy);
          maxY = Math.max(maxY, cy);
          genomeCounts.set(genomeId?.[ci] || 0, (genomeCounts.get(genomeId?.[ci] || 0) || 0) + 1);
          lineageCounts.set(lineageId?.[ci] || 0, (lineageCounts.get(lineageId?.[ci] || 0) || 0) + 1);
          for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const ni = idx(cx + dx, cy + dy);
            if (!visited[ni] && alive[ni]) stack.push([cx + dx, cy + dy]);
          }
        }

        const mass = cells.length;
        components.push({
          cells,
          mass,
          centroid: { x: sx / mass, y: sy / mass },
          boundingBox: { minX, maxX, minY, maxY },
          meanAge: ageSum / mass,
          meanEnergy: energySum / mass,
          meanStress: stressSum / mass,
          dominantGenomeId: dominantKey(genomeCounts),
          dominantLineageId: dominantKey(lineageCounts),
          genomeDiversity: genomeCounts.size,
          lineageDiversity: lineageCounts.size,
        });
      }
    }
    return components;
  }

  matchScore(prev, component) {
    const distance = Math.hypot(prev.centroid.x - component.centroid.x, prev.centroid.y - component.centroid.y);
    const massDelta = Math.abs((prev.mass || 0) - component.mass) * 0.08;
    const lineagePenalty = prev.dominantLineageId === component.dominantLineageId ? 0 : 2.2;
    return distance + massDelta + lineagePenalty;
  }

  buildOrganism({ id, component, snapshot, metrics, previous, parentIds, genomeRegistry }) {
    const ageFrames = previous ? previous.ageFrames + 1 : 1;
    const centroidHistory = previous ? [...previous.centroidHistory, component.centroid] : [component.centroid];
    const massHistory = previous ? [...previous.massHistory, component.mass] : [component.mass];
    while (centroidHistory.length > this.maxHistory) centroidHistory.shift();
    while (massHistory.length > this.maxHistory) massHistory.shift();
    const genome = genomeRegistry?.get(component.dominantGenomeId);
    const stabilityScore = clamp01(1 - component.meanStress + (genome?.traits.memoryAffinity || 0.4) * 0.18);
    const fitnessScore = clamp01(component.meanEnergy * 0.45 + stabilityScore * 0.35 + Math.log2(component.mass + 1) / 8);
    const noveltyScore = clamp01(component.genomeDiversity / 8 + component.lineageDiversity / 10 + (metrics.novelty || 0) * 0.35);
    const threatScore = clamp01(component.meanStress + component.mass / 180 + (metrics.collapseRisk || 0) * 0.2);
    const organism = {
      id,
      birthGeneration: previous?.birthGeneration ?? snapshot.generation,
      deathGeneration: null,
      ageFrames,
      parentIds: parentIds || [],
      childrenIds: previous?.childrenIds || [],
      centroid: component.centroid,
      centroidHistory,
      mass: component.mass,
      massHistory,
      boundingBox: component.boundingBox,
      dominantGenomeId: component.dominantGenomeId,
      dominantLineageId: component.dominantLineageId,
      meanEnergy: component.meanEnergy,
      meanStress: component.meanStress,
      meanAge: component.meanAge,
      fitnessScore,
      noveltyScore,
      threatScore,
      stabilityScore,
      reproductionCount: previous?.reproductionCount || 0,
      regimeImpact: metrics.regime || 'CALIBRATING',
      status: 'active',
    };
    return organism;
  }

  updateLineageGraph() {
    for (const organism of this.active.values()) {
      const id = organism.dominantLineageId;
      if (!this.lineageGraph.has(id)) this.lineageGraph.set(id, { id, organismIds: new Set(), maxMass: 0, maxAge: 0 });
      const node = this.lineageGraph.get(id);
      node.organismIds.add(organism.id);
      node.maxMass = Math.max(node.maxMass, organism.mass);
      node.maxAge = Math.max(node.maxAge, organism.ageFrames);
    }
  }

  getActiveOrganisms() {
    return [...this.active.values()];
  }

  getDeadOrganisms() {
    return this.dead.slice(-256);
  }

  getEvents() {
    return this.lastEvents;
  }

  getLineageGraph() {
    return [...this.lineageGraph.values()].map(node => ({
      id: node.id,
      organismIds: [...node.organismIds],
      maxMass: node.maxMass,
      maxAge: node.maxAge,
    }));
  }

  getEventSummary(totalGenerations = 0) {
    const per1k = {};
    const denom = Math.max(1, totalGenerations / 1000);
    for (const [k,v] of Object.entries(this.confirmedEventTypeCounts)) per1k[`${k}_per_1k`] = v / denom;
    return { confirmedEventTypeCounts: this.confirmedEventTypeCounts, candidateEventTypeCounts: this.candidateEventTypeCounts, eventRatesPer1kGenerations: per1k, recentConfirmedEvents: this.lastEvents.slice(-50), recentCandidateEvents: this.recentCandidateEvents.slice(-50) };
  }

  exportState() {
    return {
      active: this.getActiveOrganisms(),
      dead: this.getDeadOrganisms(),
      lineages: this.getLineageGraph(),
      nextId: this.nextId,
    };
  }
}

function dominantKey(counts) {
  let best = 0;
  let count = -1;
  for (const [key, value] of counts) {
    if (value > count) {
      best = key;
      count = value;
    }
  }
  return best;
}

function event(generation, type, severity, organism, description) {
  return {
    generation,
    type,
    severity,
    organismId: organism.id,
    lineageId: organism.dominantLineageId,
    genomeId: organism.dominantGenomeId,
    description,
  };
}
