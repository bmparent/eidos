export class EvolutionTelemetry {
  constructor() {
    this.prevDominantGenomeId = 0;
    this.prevActiveLineages = new Set();
    this.events = [];
    this.noveltyArchive = new Set();
  }

  reset() {
    this.prevDominantGenomeId = 0;
    this.prevActiveLineages.clear();
    this.events = [];
    this.noveltyArchive.clear();
  }

  record({ engine, organisms, organismEvents, localRegimes, prediction, metrics }) {
    const genomeCounts = new Map();
    const lineageCounts = new Map();
    let living = 0;
    for (let i = 0; i < engine.size; i++) {
      if (!engine.alive[i]) continue;
      living++;
      genomeCounts.set(engine.genomeId[i], (genomeCounts.get(engine.genomeId[i]) || 0) + 1);
      lineageCounts.set(engine.lineageId[i], (lineageCounts.get(engine.lineageId[i]) || 0) + 1);
    }

    const dominantGenomeId = dominant(genomeCounts);
    const dominantLineageId = dominant(lineageCounts);
    const activeLineages = new Set([...lineageCounts.keys()].filter(Boolean));
    const extinct = [...this.prevActiveLineages].filter(id => !activeLineages.has(id));
    const longLived = organisms.find(organism => organism.ageFrames > 0 && organism.ageFrames % 600 === 0);
    const genomeDiversity = diversityIndex(genomeCounts, living);
    const lineageDiversity = diversityIndex(lineageCounts, living);
    const evolutionEvents = [...organismEvents];

    if (dominantGenomeId && this.prevDominantGenomeId && dominantGenomeId !== this.prevDominantGenomeId) {
      evolutionEvents.push({
        generation: engine.generation,
        type: 'dominant_genome_shift',
        severity: 'medium',
        genomeId: dominantGenomeId,
        description: `Dominant genome shifted: G${this.prevDominantGenomeId} -> G${dominantGenomeId}.`,
      });
    }
    for (const lineageId of extinct) {
      evolutionEvents.push({
        generation: engine.generation,
        type: 'lineage_extinction',
        severity: 'medium',
        lineageId,
        description: `Lineage #${lineageId} went extinct after losing all active cells.`,
      });
    }
    if (longLived) {
      evolutionEvents.push({
        generation: engine.generation,
        type: 'long_lived_organism',
        severity: 'medium',
        organismId: longLived.id,
        lineageId: longLived.dominantLineageId,
        genomeId: longLived.dominantGenomeId,
        description: `Long-lived organism detected: #${longLived.id}, ${longLived.ageFrames} frames.`,
      });
    }
    if (localRegimes?.lastVioletCount > 0 && metrics.regime !== 'VIOLET') {
      evolutionEvents.push({
        generation: engine.generation,
        type: 'violet_zone_emergence',
        severity: 'medium',
        description: `${localRegimes.lastVioletCount} local VIOLET zone(s) emerged inside a ${metrics.regime} global regime.`,
      });
    }
    if ((prediction?.predictionError || 0) > 0.18) {
      evolutionEvents.push({
        generation: engine.generation,
        type: 'high_prediction_error',
        severity: 'medium',
        description: `Prediction error rose to ${prediction.predictionError.toFixed(3)}.`,
      });
    }

    for (const event of evolutionEvents) this.events.push(event);
    if (this.events.length > 500) this.events.splice(0, this.events.length - 500);
    this.prevDominantGenomeId = dominantGenomeId;
    this.prevActiveLineages = activeLineages;

    const oldestOrganismAge = organisms.reduce((max, organism) => Math.max(max, organism.ageFrames), 0);
    const largestOrganismMass = organisms.reduce((max, organism) => Math.max(max, organism.mass), 0);
    const metricsOut = {
      livingLineages: activeLineages.size,
      totalLineages: engine.genomeRegistry.lineages.size,
      activeGenomes: genomeCounts.size,
      dominantGenomeId,
      dominantLineageId,
      oldestOrganismAge,
      largestOrganismMass,
      genomeDiversity,
      speciesDiversity: genomeDiversity,
      lineageDiversity,
      birthRate: engine.lastBirthCount / engine.size,
      deathRate: engine.lastDeathCount / engine.size,
      splitRate: organismEvents.filter(event => event.type === 'organism_split').length,
      mergeRate: organismEvents.filter(event => event.type === 'organism_merge').length,
      mutationRate: engine.lastBirthCount ? engine.lastMutationCount / engine.lastBirthCount : 0,
      extinctionCount: extinct.length,
      noveltyArchiveSize: this.noveltyArchive.size,
      stableStructureCount: organisms.filter(organism => organism.stabilityScore > 0.76 && organism.ageFrames > 90).length,
      localRegimeDiversity: localRegimes?.diversity?.() || 1,
      predictionError: prediction?.predictionError || 0,
    };

    return { metrics: metricsOut, events: evolutionEvents };
  }

  exportData({ engine, organismTracker } = {}) {
    return {
      genomes: engine?.genomeRegistry.exportGenomes() || [],
      lineages: engine?.genomeRegistry.exportLineages() || [],
      organisms: organismTracker?.getActiveOrganisms() || [],
      deadOrganisms: organismTracker?.getDeadOrganisms() || [],
      lineageGraph: organismTracker?.getLineageGraph() || [],
      events: this.events,
    };
  }
}

function dominant(counts) {
  let best = 0;
  let bestCount = -1;
  for (const [id, count] of counts) {
    if (count > bestCount) {
      best = id;
      bestCount = count;
    }
  }
  return best;
}

function diversityIndex(counts, total) {
  if (!total) return 0;
  let sum = 0;
  for (const count of counts.values()) {
    const p = count / total;
    sum += p * p;
  }
  return 1 - sum;
}
